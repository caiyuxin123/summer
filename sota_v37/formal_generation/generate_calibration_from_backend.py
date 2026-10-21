from __future__ import annotations
"""Formal 120-scenario orchestrator.

The physics backend is integrity-gated.  It must first reproduce the frozen Feb
Reference/Direct-M facts before this script can emit calibration_120.npz.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import numpy as np
from .apply_scenario import perturb_base_load

REQUIRED_KEYS = {"x_month", "x_history", "graph", "y"}
_WORKER_BACKEND = None


def load_backend(path):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("paper_physics_backend_dynamic", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    for name in ("FROZEN_FEB_VALIDATED", "build_calibration_sample"):
        if not hasattr(m, name):
            raise AttributeError(f"backend missing {name}")
    if m.FROZEN_FEB_VALIDATED is not True:
        raise RuntimeError("Physics backend has not passed frozen-Feb reproduction; refusing formal calibration generation.")
    return m


def _init_worker(backend_path: str):
    global _WORKER_BACKEND
    # Prevent solver/BLAS oversubscription when several scenario workers run in parallel.
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    _WORKER_BACKEND = load_backend(backend_path)


def _one(args):
    idx, base, row = args
    target = perturb_base_load(base, row, "target")
    context = perturb_base_load(base, row, "context")
    s = _WORKER_BACKEND.build_calibration_sample(target, context)
    if set(s) != REQUIRED_KEYS:
        raise ValueError(f"backend keys {set(s)} != {REQUIRED_KEYS}")
    return idx, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-history-npy", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=1)
    a = p.parse_args()

    base = np.load(a.base_history_npy)
    if base.shape != (168, 3):
        raise ValueError(f"base history must be (168,3), got {base.shape}")
    rows = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    if len(rows) != 120:
        raise ValueError("manifest must contain exactly 120 scenarios")

    # Fail fast in the parent before creating workers.
    load_backend(a.backend)
    tasks = [(i, base, r) for i, r in enumerate(rows)]
    if int(a.jobs) <= 1:
        _init_worker(a.backend)
        got = [_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=int(a.jobs), initializer=_init_worker, initargs=(str(Path(a.backend).resolve()),)) as ex:
            got = list(ex.map(_one, tasks, chunksize=1))
    got.sort(key=lambda z: z[0])
    samples = [s for _, s in got]

    graph = np.asarray(samples[0]["graph"], dtype=np.float32)
    for s in samples:
        if not np.allclose(s["graph"], graph, atol=0.0, rtol=0.0):
            raise ValueError("graph differs across calibration scenarios")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        x_month=np.stack([s["x_month"] for s in samples]).astype(np.float32),
        x_history=np.stack([s["x_history"] for s in samples]).astype(np.float32),
        graph=graph,
        y=np.stack([s["y"] for s in samples]).astype(np.float32),
    )


if __name__ == "__main__":
    main()
