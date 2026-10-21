from __future__ import annotations
"""Exact SOTA feature definitions frozen before formal target evaluation.

The definitions are carried forward from the 2026-09-29 SOTA-input-freeze package.
Only target-period coarse aggregates enter x_month; x_history is generated from an
independent 168 h history window.
"""
import numpy as np
import pandas as pd

HISTORY_CHANNELS = [
    "load_pu",
    "thermal_generation_pu",
    "zero_carbon_generation_pu",
    "net_export_pu",
    "local_carbon_intensity_1000",
]

MONTHLY_FEATURE_NAMES = [
    "area1_avg_load_pu", "area2_avg_load_pu", "area3_avg_load_pu",
]


def area_graph(net) -> np.ndarray:
    """3-area symmetric weighted graph from inter-area continuous ratings."""
    areas = net.areas
    idx = {a: i for i, a in enumerate(areas)}
    A = np.zeros((len(areas), len(areas)), dtype=float)
    for l in range(len(net.rating)):
        af = int(net.area_of_bus[net.f_idx[l]])
        at = int(net.area_of_bus[net.t_idx[l]])
        if af == at:
            continue
        i, j = idx[af], idx[at]
        A[i, j] += float(net.rating[l])
        A[j, i] += float(net.rating[l])
    A = A + np.eye(len(areas)) * max(A.max(), 1.0) * 0.1
    d = A.sum(axis=1)
    D = np.diag(np.where(d > 0, d ** -0.5, 0.0))
    return D @ A @ D


def category_schema(gen: pd.DataFrame) -> list[tuple[int, str, np.ndarray, float]]:
    rows = []
    for a in sorted(gen.area.astype(int).unique()):
        for typ in sorted(gen.loc[gen.area == a, "typ"].unique()):
            ids = np.where((gen.area.to_numpy(int) == a) & (gen.typ.to_numpy() == typ))[0]
            rows.append((int(a), str(typ), ids, float(gen.pmax.to_numpy(float)[ids].sum())))
    return rows


def monthly_feature_names(gen: pd.DataFrame, areas: list[int]) -> list[str]:
    names = list(MONTHLY_FEATURE_NAMES)
    for a, typ, _, _ in category_schema(gen):
        names.append(f"gen_cf_A{a}_{typ}")
    for a in areas:
        for b in areas:
            if a != b:
                names.append(f"directed_flow_A{a}_to_A{b}_share")
    for a in areas:
        names.append(f"direct_cef_A{a}_1000")
    return names


def area_base_load_mw(net) -> np.ndarray:
    return np.asarray([
        net.bus.loc[net.bus["Area"].astype(int) == a, "MW Load"].astype(float).sum()
        for a in net.areas
    ], dtype=float)


def _area_capacities(gen: pd.DataFrame, areas: list[int]) -> tuple[np.ndarray, np.ndarray]:
    thermal = np.zeros(len(areas))
    zero = np.zeros(len(areas))
    for j, a in enumerate(areas):
        ids = np.where(gen.area.to_numpy(int) == a)[0]
        efs = gen.ef.to_numpy(float)[ids]
        cap = gen.pmax.to_numpy(float)[ids]
        thermal[j] = cap[efs > 1e-12].sum()
        zero[j] = cap[efs <= 1e-12].sum()
    return thermal, zero


def history_tensor(net, gen, Lreg: np.ndarray, P: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Return T x Area x 5 tensor using only independent-history information."""
    T, C = Lreg.shape
    area_base = np.maximum(area_base_load_mw(net), 1.0)
    thermal_cap, zero_cap = _area_capacities(gen, net.areas)
    thermal_cap = np.maximum(thermal_cap, 1.0)
    zero_cap = np.maximum(zero_cap, 1.0)

    thermal = np.zeros((T, C))
    zero = np.zeros((T, C))
    cinj = np.zeros((T, C))
    for j, a in enumerate(net.areas):
        ids = np.where(gen.area.to_numpy(int) == a)[0]
        efs = gen.ef.to_numpy(float)[ids]
        thermal[:, j] = P[:, ids][:, efs > 1e-12].sum(axis=1)
        zero[:, j] = P[:, ids][:, efs <= 1e-12].sum(axis=1)
        cinj[:, j] = (P[:, ids] * efs[None, :]).sum(axis=1)

    area_idx = {a: i for i, a in enumerate(net.areas)}
    outgoing = np.zeros((T, C))
    incoming = np.zeros((T, C))
    tie_rating = np.zeros(C)
    for l in range(len(net.rating)):
        af = int(net.area_of_bus[net.f_idx[l]])
        at = int(net.area_of_bus[net.t_idx[l]])
        if af == at:
            continue
        i, j = area_idx[af], area_idx[at]
        f = F[:, l]
        outgoing[:, i] += np.maximum(f, 0)
        incoming[:, j] += np.maximum(f, 0)
        outgoing[:, j] += np.maximum(-f, 0)
        incoming[:, i] += np.maximum(-f, 0)
        tie_rating[i] += net.rating[l]
        tie_rating[j] += net.rating[l]
    tie_rating = np.maximum(tie_rating, 1.0)

    H = np.zeros((T, C, len(HISTORY_CHANNELS)))
    H[:, :, 0] = Lreg / area_base[None, :]
    H[:, :, 1] = thermal / thermal_cap[None, :]
    H[:, :, 2] = zero / zero_cap[None, :]
    H[:, :, 3] = (outgoing - incoming) / tie_rating[None, :]
    H[:, :, 4] = cinj / np.maximum(Lreg, 1e-6) / 1000.0
    return H.astype(np.float32)


def monthly_features(net, gen, Lreg: np.ndarray, P: np.ndarray, state) -> np.ndarray:
    """Allowed target-period coarse features only; no target-hour state is returned."""
    T = len(Lreg)
    total_energy = float(Lreg.sum())
    area_base = area_base_load_mw(net)
    feats = list((Lreg.mean(axis=0) / np.maximum(area_base, 1.0)).astype(float))
    for _a, _typ, ids, cap in category_schema(gen):
        energy = float(P[:, ids].sum())
        feats.append(energy / max(T * cap, 1e-9))
    af = state.area_flow.sum(axis=0)
    for i, a in enumerate(net.areas):
        for j, b in enumerate(net.areas):
            if i != j:
                feats.append(float(af[i, j] / max(total_energy, 1e-9)))
    feats.extend((state.direct_month / 1000.0).astype(float).tolist())
    arr = np.asarray(feats, dtype=np.float32)
    if arr.shape != (34,):
        raise RuntimeError(f"Frozen monthly feature dimension changed: {arr.shape}")
    return arr
