from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog
from .network import DCNetwork, generator_incidence


def dc_opf_timeseries(net: DCNetwork, gen: pd.DataFrame, nodal_load: np.ndarray, method: str = "highs") -> tuple[np.ndarray,np.ndarray]:
    """Linear DC OPF, no UC/min-output. Returns P[T,G], F[T,L]."""
    Cg = generator_incidence(net, gen)
    Hg = net.H @ Cg
    T = len(nodal_load); G = len(gen); L = len(net.rating)
    P = np.zeros((T,G)); F = np.zeros((T,L))
    pmax = gen.pmax.to_numpy(float); cost = gen.cost.to_numpy(float)
    Aub = np.vstack([Hg, -Hg])
    bounds = list(zip(np.zeros(G), pmax))
    for t, load in enumerate(nodal_load):
        hload = net.H @ load
        bub = np.r_[net.rating + hload, net.rating - hload]
        res = linprog(cost, A_ub=Aub, b_ub=bub,
                      A_eq=np.ones((1,G)), b_eq=[load.sum()],
                      bounds=bounds, method=method)
        if not res.success:
            raise RuntimeError(f"DC OPF failed at t={t}: {res.message}")
        P[t] = res.x
        F[t] = Hg @ res.x - hload
    return P,F


def dc_opf_timeseries_variable_pmax(net: DCNetwork, gen: pd.DataFrame, nodal_load: np.ndarray,
                                     pmax_time: np.ndarray, method: str="highs") -> tuple[np.ndarray,np.ndarray]:
    """DC OPF with time-varying generator upper bounds (used for wind availability)."""
    Cg = generator_incidence(net, gen)
    Hg = net.H @ Cg
    T = len(nodal_load); G = len(gen); L = len(net.rating)
    if pmax_time.shape != (T,G):
        raise ValueError(f"pmax_time must be {(T,G)}, got {pmax_time.shape}")
    P=np.zeros((T,G)); F=np.zeros((T,L)); cost=gen.cost.to_numpy(float)
    Aub=np.vstack([Hg,-Hg])
    for t,load in enumerate(nodal_load):
        hload=net.H@load
        bub=np.r_[net.rating+hload, net.rating-hload]
        res=linprog(cost,A_ub=Aub,b_ub=bub,
                    A_eq=np.ones((1,G)),b_eq=[load.sum()],
                    bounds=list(zip(np.zeros(G),pmax_time[t])),method=method)
        if not res.success:
            raise RuntimeError(f"Variable-pmax DC OPF failed at t={t}: {res.message}")
        P[t]=res.x; F[t]=Hg@res.x-hload
    return P,F
