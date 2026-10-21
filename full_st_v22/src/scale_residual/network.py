from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.linalg import lu_factor, lu_solve

@dataclass
class DCNetwork:
    bus: pd.DataFrame
    branch: pd.DataFrame
    bus_ids: list[int]
    bus_index: dict[int,int]
    areas: list[int]
    area_of_bus: np.ndarray
    load_share: np.ndarray
    H: np.ndarray
    rating: np.ndarray
    f_idx: np.ndarray
    t_idx: np.ndarray


def build_dc_network(bus: pd.DataFrame, branch: pd.DataFrame, base_mva: float = 100.0, slack_bus: int | None = None) -> DCNetwork:
    bus = bus.copy(); branch = branch.copy()
    bus_ids = bus["Bus ID"].astype(int).tolist()
    idx = {b:i for i,b in enumerate(bus_ids)}
    n = len(bus); nl = len(branch)
    f_idx = branch["From Bus"].astype(int).map(idx).to_numpy()
    t_idx = branch["To Bus"].astype(int).map(idx).to_numpy()
    x = branch["X"].astype(float).to_numpy()
    if np.any(np.abs(x) < 1e-12):
        raise ValueError("Zero-reactance branch encountered")
    bcoef = base_mva / x
    inc = np.zeros((nl,n))
    inc[np.arange(nl), f_idx] = 1.0
    inc[np.arange(nl), t_idx] = -1.0
    B = inc.T @ (bcoef[:,None] * inc)
    if slack_bus is None:
        slack_bus = bus_ids[0]
    s = idx[int(slack_bus)]
    keep = np.array([i for i in range(n) if i != s], dtype=int)
    lu, piv = lu_factor(B[np.ix_(keep,keep)])
    inv = lu_solve((lu,piv), np.eye(n-1))
    H = np.zeros((nl,n))
    H[:,keep] = (bcoef[:,None] * inc[:,keep]) @ inv
    areas = sorted(bus["Area"].astype(int).unique().tolist())
    area_of_bus = bus["Area"].astype(int).to_numpy()
    base_load = bus["MW Load"].astype(float).to_numpy()
    share = np.zeros(n)
    for a in areas:
        ids = np.where(area_of_bus == a)[0]
        den = base_load[ids].sum()
        if den <= 0:
            raise ValueError(f"Area {a} has zero base load; cannot form static load shares")
        share[ids] = base_load[ids] / den
    rating_col = "Cont Rating" if "Cont Rating" in branch.columns else "Rate A"
    rating = branch[rating_col].astype(float).to_numpy()
    return DCNetwork(bus, branch, bus_ids, idx, areas, area_of_bus, share, H, rating, f_idx, t_idx)


def generator_incidence(net: DCNetwork, gen: pd.DataFrame) -> np.ndarray:
    Cg = np.zeros((len(net.bus_ids), len(gen)))
    for k,r in gen.iterrows():
        Cg[net.bus_index[int(r.bus)], k] = 1.0
    return Cg


def regional_to_nodal_load(net: DCNetwork, regional: np.ndarray) -> np.ndarray:
    """regional: T x C in net.areas order -> T x N nodal loads."""
    regional = np.asarray(regional, dtype=float)
    T = regional.shape[0]
    out = np.zeros((T, len(net.bus_ids)))
    for j,a in enumerate(net.areas):
        ids = np.where(net.area_of_bus == a)[0]
        out[:,ids] = regional[:,j,None] * net.load_share[ids][None,:]
    return out
