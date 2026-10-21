from __future__ import annotations
import numpy as np
import pandas as pd
from .network import DCNetwork, generator_incidence


def nodal_cef(net: DCNetwork, gen: pd.DataFrame, p: np.ndarray, load: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Solve nodal carbon emission factor using directed proportional sharing.

    Units: p/load/flow in MW; gen.ef in kgCO2/MWh; returns kgCO2/MWh.
    """
    Cg = generator_incidence(net, gen)
    carbon = Cg @ (p * gen.ef.to_numpy(float))
    local_gen = Cg @ p
    n = len(net.bus_ids)
    A = np.zeros((n,n)); throughput = local_gen.copy()
    for l, f in enumerate(flow):
        i = net.f_idx[l]; j = net.t_idx[l]
        if f >= 0:
            throughput[j] += f
            A[j,i] -= f
        else:
            q = -f
            throughput[i] += q
            A[i,j] -= q
    A[np.arange(n),np.arange(n)] += throughput
    z = np.where(throughput < 1e-10)[0]
    for i in z:
        A[i,:] = 0.0; A[i,i] = 1.0; carbon[i] = 0.0
    try:
        return np.linalg.solve(A, carbon)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, carbon, rcond=None)[0]


def nodal_cef_timeseries(net: DCNetwork, gen: pd.DataFrame, P: np.ndarray, Lbus: np.ndarray, F: np.ndarray) -> np.ndarray:
    return np.vstack([nodal_cef(net,gen,P[t],Lbus[t],F[t]) for t in range(len(P))])
