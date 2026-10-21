from __future__ import annotations
"""Audited RTS-GMLC v8 physics backend for SOTA calibration generation.

The backend uses the exact v2.2 manuscript engine core and frozen raw inputs.  The
formal generator is enabled only when a frozen-Feb reproduction gate exists and
its source/code hashes still match the files in this package.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from paper_physics.network import build_dc_network, regional_to_nodal_load
from paper_physics.generator_profile import build_paper_compat_profile, aggregate_bus_type
from paper_physics.dispatch import dc_opf_timeseries
from paper_physics.carbon import nodal_cef_timeseries
from paper_physics.aggregate import aggregate_state
from paper_physics.features import (
    area_graph, history_tensor, monthly_features, monthly_feature_names,
    category_schema, area_base_load_mw,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_physics" / "data"
GATE_PATH = ROOT / "formal_generation" / "artifacts" / "FROZEN_FEB_GATE.json"
CORE_FILES = [
    ROOT / "paper_physics" / "network.py",
    ROOT / "paper_physics" / "dispatch.py",
    ROOT / "paper_physics" / "carbon.py",
    ROOT / "paper_physics" / "aggregate.py",
    ROOT / "paper_physics" / "generator_profile.py",
    ROOT / "paper_physics" / "features.py",
    Path(__file__),
]
RAW_FILES = [
    DATA / "bus.csv", DATA / "branch.csv", DATA / "gen.csv",
    DATA / "DAY_AHEAD_regional_Load.csv",
]

EXPECTED_RAW_SHA256 = {
    "bus.csv": "cec3f776222812d43eaeaf7ac85b577719dc565f354a04e120c12a647ddb710e",
    "branch.csv": "e92d16d13b1c5d2899f7871c0c99c3eb94605528beef157bf9e1ccbcffe5ef4e",
    "gen.csv": "23a5203d77920110ae70919feb19bb6c6175208a4d22767e1d30bfa871b00a8f",
    "DAY_AHEAD_regional_Load.csv": "5a76a7ffe673c6531a10e83556d313eb028dff96e4209166909e74bf4e39b254",
}

# Full-precision frozen v8/v2.2 facts. Tolerances are deliberately much tighter
# than the 3-decimal values reported in the paper.
EXPECTED_FEB = {
    "fine_month": [899.9342502039029, 935.7356554958692, 450.47778855774783],
    "hourly_coarse_month": [689.3144996412555, 907.668251261296, 632.3657635099516],
    "direct_month": [688.9463394431672, 908.088397214504, 632.3215190796625],
    "delta_spatial": [210.61975056264748, 28.06740423457302, -181.88797495220385],
    "delta_temporal": [0.368160198088276, -0.420145953207907, 0.0442444302891544],
    "delta_total": [210.98791076073576, 27.64725828136511, -181.8437305219147],
    "area_energy_mwh": [767495.763247, 778776.7432716, 1008906.762105],
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def required_hashes() -> dict[str, str]:
    out = {}
    for p in RAW_FILES + CORE_FILES:
        out[str(p.relative_to(ROOT))] = _sha256(p)
    return out


def validate_raw_hashes() -> None:
    for name, expected in EXPECTED_RAW_SHA256.items():
        got = _sha256(DATA / name)
        if got != expected:
            raise RuntimeError(f"Frozen raw input hash mismatch for {name}: {got} != {expected}")


@lru_cache(maxsize=1)
def prepared():
    validate_raw_hashes()
    bus = pd.read_csv(DATA / "bus.csv")
    branch = pd.read_csv(DATA / "branch.csv")
    net = build_dc_network(bus, branch)
    gen = aggregate_bus_type(build_paper_compat_profile(), bus)
    return bus, branch, net, gen


def read_loads() -> pd.DataFrame:
    df = pd.read_csv(DATA / "DAY_AHEAD_regional_Load.csv")
    dt = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day)) + pd.to_timedelta(df.Period - 1, unit="h")
    out = df.copy()
    out.index = pd.DatetimeIndex(dt, name="timestamp")
    return out


def load_window(start: str, end: str) -> np.ndarray:
    df = read_loads()
    s = pd.Timestamp(start)
    e = pd.Timestamp(end) + pd.Timedelta(hours=23)
    sub = df.loc[(df.index >= s) & (df.index <= e), ["1", "2", "3"]]
    return sub.to_numpy(float)


def load_month(year: int, month: int) -> np.ndarray:
    df = read_loads()
    sub = df.loc[(df.index.year == int(year)) & (df.index.month == int(month)), ["1", "2", "3"]]
    return sub.to_numpy(float)


def feb2020_weekday_hour_indices() -> np.ndarray:
    hist_times = pd.date_range("2020-01-01 00:00", "2020-01-07 23:00", freq="h")
    feb_times = pd.date_range("2020-02-01 00:00", "2020-02-29 23:00", freq="h")
    lookup = {(t.weekday(), t.hour): i for i, t in enumerate(hist_times)}
    idx = np.asarray([lookup[(t.weekday(), t.hour)] for t in feb_times], dtype=int)
    if idx.shape != (696,):
        raise RuntimeError(idx.shape)
    return idx


def expand_week_template_to_feb2020(template_168x3: np.ndarray) -> np.ndarray:
    """Map a synthetic Jan-1--7 weekly template to all 696 h of Feb 2020.

    Calendar labels only are used.  The 168 h values themselves come entirely from
    the independently frozen history pool plus the already-frozen perturbation manifest.
    This duration alignment prevents a 168 h pseudo-month from being used as the training
    target for a 696 h formal monthly task.
    """
    x = np.asarray(template_168x3, dtype=float)
    if x.shape != (168, 3):
        raise ValueError(f"weekly template must be (168,3), got {x.shape}")
    return x[feb2020_weekday_hour_indices()]


def _run_state(Lreg: np.ndarray, need_cef: bool = True):
    _bus, _branch, net, gen = prepared()
    Lreg = np.asarray(Lreg, dtype=float)
    if Lreg.ndim != 2 or Lreg.shape[1] != 3 or np.any(Lreg <= 0) or not np.isfinite(Lreg).all():
        raise ValueError(f"invalid regional load array: {Lreg.shape}")
    Lbus = regional_to_nodal_load(net, Lreg)
    P, F = dc_opf_timeseries(net, gen, Lbus, method="highs")
    if not need_cef:
        return Lbus, P, F, None
    X = nodal_cef_timeseries(net, gen, P, Lbus, F)
    state = aggregate_state(net, gen, Lreg, Lbus, P, F, X)
    return Lbus, P, F, state


def fresh_feb_reproduction() -> dict:
    _bus, _branch, net, gen = prepared()
    L = load_month(2020, 2)
    Lbus, P, F, state = _run_state(L, need_cef=True)
    Cg = np.zeros((len(net.bus_ids), len(gen)))
    for k, r in gen.iterrows():
        Cg[net.bus_index[int(r.bus)], k] = 1.0
    # power balance is checked from generation/load equality; max line loading is direct.
    balance = np.max(np.abs(P.sum(axis=1) - Lbus.sum(axis=1)))
    flow_ratio = np.max(np.abs(F) / net.rating[None, :])
    return {
        "fine_month": state.fine_month.tolist(),
        "hourly_coarse_month": state.hourly_coarse_month.tolist(),
        "direct_month": state.direct_month.tolist(),
        "delta_spatial": state.delta_spatial.tolist(),
        "delta_temporal": state.delta_temporal.tolist(),
        "delta_total": state.delta_total.tolist(),
        "closure": state.closure.tolist(),
        "area_energy_mwh": L.sum(axis=0).tolist(),
        "max_flow_ratio": float(flow_ratio),
        "max_power_balance_mw": float(balance),
        "dimensions": {
            "buses": len(net.bus_ids), "branches": len(net.branch),
            "dispatch_units": len(gen), "area_type_categories": len(category_schema(gen)),
            "hours": len(L),
        },
        "area_base_load_mw": area_base_load_mw(net).tolist(),
        "feature_names": monthly_feature_names(gen, net.areas),
    }


def compare_feb(result: dict, tol: float = 1e-8) -> dict:
    diffs = {}
    passed = True
    for k, expected in EXPECTED_FEB.items():
        got = np.asarray(result[k], dtype=float)
        exp = np.asarray(expected, dtype=float)
        m = float(np.max(np.abs(got - exp)))
        diffs[k] = m
        passed &= m <= tol
    closure = float(np.max(np.abs(result["closure"])))
    passed &= closure <= 1e-10
    passed &= result["dimensions"] == {
        "buses": 73, "branches": 120, "dispatch_units": 45,
        "area_type_categories": 22, "hours": 696,
    }
    passed &= float(result["max_power_balance_mw"]) <= 1e-8
    passed &= float(result["max_flow_ratio"]) <= 1.0 + 1e-10
    return {"passed": bool(passed), "max_abs_differences": diffs, "closure_max": closure, "tolerance": tol}


def write_frozen_gate(path: Path = GATE_PATH) -> dict:
    validate_raw_hashes()
    result = fresh_feb_reproduction()
    comparison = compare_feb(result)
    payload = {
        "gate_version": "2026-09-29-v3.3-v8-frozen-feb",
        "validated": bool(comparison["passed"]),
        "source_baseline": "paper v8 + reproduction engine v2.2",
        "fresh_reproduction": result,
        "comparison": comparison,
        "required_hashes": required_hashes(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not payload["validated"]:
        raise RuntimeError("Frozen-Feb reproduction gate FAILED")
    return payload


def _gate_is_valid() -> bool:
    try:
        if not GATE_PATH.exists():
            return False
        g = json.loads(GATE_PATH.read_text(encoding="utf-8"))
        if g.get("validated") is not True:
            return False
        if g.get("required_hashes") != required_hashes():
            return False
        # Do not trust a stale copied gate whose frozen comparison failed.
        if g.get("comparison", {}).get("passed") is not True:
            return False
        return True
    except Exception:
        return False


def build_calibration_sample(target_168x3: np.ndarray, context_168x3: np.ndarray) -> dict:
    if not _gate_is_valid():
        raise RuntimeError("Frozen-Feb gate is missing/stale; formal calibration generation is disabled")
    _bus, _branch, net, gen = prepared()
    target_week = np.asarray(target_168x3, dtype=float)
    context_week = np.asarray(context_168x3, dtype=float)
    if target_week.shape != (168,3) or context_week.shape != (168,3):
        raise ValueError("target/context templates must both be (168,3)")

    # The 696 h synthetic month is a deterministic weekday-hour reuse of the 168 h
    # target template.  DC-OPF/CEF are memoryless in the frozen manuscript engine, so
    # solve the 168 unique hourly states once and expand their states by the same index.
    # This is algebraically identical to solving all 696 repeated hours and cuts formal
    # calibration generation time by ~2.5x without changing any label.
    idx = feb2020_weekday_hour_indices()
    Lbus_w, P_w, F_w, _ = _run_state(target_week, need_cef=False)
    X_w = nodal_cef_timeseries(net, gen, P_w, Lbus_w, F_w)
    target_month = target_week[idx]
    Lbus_m = Lbus_w[idx]
    P_m = P_w[idx]
    F_m = F_w[idx]
    X_m = X_w[idx]
    state = aggregate_state(net, gen, target_month, Lbus_m, P_m, F_m, X_m)

    _Lbc, Pc, Fc, _ = _run_state(context_week, need_cef=False)
    return {
        "x_month": monthly_features(net, gen, target_month, P_m, state).astype(np.float32),
        "x_history": history_tensor(net, gen, context_week, Pc, Fc).astype(np.float32),
        "graph": area_graph(net).astype(np.float32),
        "y": (state.fine_month / 1000.0).astype(np.float32),
    }


def build_formal_feb_input() -> tuple[dict, np.ndarray, dict]:
    """Build input-only formal Feb features and a separately returned Reference y.

    The returned model input contains only x_month/x_history/graph. Target-hour state
    arrays are transient inside this exporter and are never written to the model-input NPZ.
    """
    if not _gate_is_valid():
        raise RuntimeError("Frozen-Feb gate is missing/stale")
    _bus, _branch, net, gen = prepared()
    feb = load_month(2020, 2)
    hist = load_window("2020-01-01", "2020-01-07")
    _Lb, P, F, state = _run_state(feb, need_cef=True)
    _Lbh, Ph, Fh, _ = _run_state(hist, need_cef=False)
    model_input = {
        "x_month": monthly_features(net, gen, feb, P, state)[None, :].astype(np.float32),
        "x_history": history_tensor(net, gen, hist, Ph, Fh)[None, ...].astype(np.float32),
        "graph": area_graph(net).astype(np.float32),
    }
    y = (state.fine_month / 1000.0)[None, :].astype(np.float32)
    audit = {
        "input_keys": sorted(model_input),
        "reference_keys": ["y"],
        "target_hour_arrays_exported": False,
        "x_month_shape": list(model_input["x_month"].shape),
        "x_history_shape": list(model_input["x_history"].shape),
        "graph_shape": list(model_input["graph"].shape),
        "reference_shape": list(y.shape),
        "direct_m_div_1000": model_input["x_month"][0, 31:34].astype(float).tolist(),
    }
    return model_input, y, audit


# This constant is recomputed from the integrity-bound gate at module import.
FROZEN_FEB_VALIDATED = _gate_is_valid()
