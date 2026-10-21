from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .network import DCNetwork

@dataclass
class ScaleOutputs:
    fine_month: np.ndarray
    hourly_coarse_month: np.ndarray
    direct_month: np.ndarray
    delta_spatial: np.ndarray
    delta_temporal: np.ndarray
    delta_total: np.ndarray
    closure: np.ndarray
    area_hour_cef: np.ndarray
    area_flow: np.ndarray


def area_direct_cef(local_generation: np.ndarray, carbon_injection: np.ndarray, directed_flow: np.ndarray) -> np.ndarray:
    """CEF on a C-area directed coarse network."""
    C = len(local_generation)
    A = np.zeros((C,C)); throughput = local_generation.astype(float).copy()
    b = carbon_injection.astype(float).copy()
    for i in range(C):
        for j in range(C):
            if i == j: continue
            q = directed_flow[j,i]  # incoming j -> i
            if q > 0:
                throughput[i] += q
                A[i,j] -= q
    A[np.arange(C),np.arange(C)] += throughput
    z = np.where(throughput < 1e-10)[0]
    for i in z:
        A[i,:] = 0; A[i,i] = 1; b[i] = 0
    return np.linalg.solve(A,b)


def aggregate_state(net: DCNetwork, gen: pd.DataFrame, regional_load: np.ndarray,
                    nodal_load: np.ndarray, P: np.ndarray, F: np.ndarray,
                    X: np.ndarray) -> ScaleOutputs:
    areas = net.areas; C = len(areas); T = len(P)
    area_idx = {a:i for i,a in enumerate(areas)}
    fine = np.zeros(C)
    for j,a in enumerate(areas):
        ids = np.where(net.area_of_bus == a)[0]
        fine[j] = (nodal_load[:,ids] * X[:,ids]).sum() / regional_load[:,j].sum()
    Parea = np.zeros((T,C)); Carea = np.zeros((T,C))
    for j,a in enumerate(areas):
        gi = np.where(gen.area.to_numpy(int)==a)[0]
        Parea[:,j] = P[:,gi].sum(axis=1)
        Carea[:,j] = (P[:,gi] * gen.ef.to_numpy(float)[gi]).sum(axis=1)
    aflow = np.zeros((T,C,C))
    for l in range(len(net.rating)):
        af = int(net.area_of_bus[net.f_idx[l]]); at = int(net.area_of_bus[net.t_idx[l]])
        if af == at: continue
        i=area_idx[af]; j=area_idx[at]
        pos=np.maximum(F[:,l],0); neg=np.maximum(-F[:,l],0)
        aflow[:,i,j] += pos
        aflow[:,j,i] += neg
    yh = np.vstack([area_direct_cef(Parea[t],Carea[t],aflow[t]) for t in range(T)])
    hourly_month = (regional_load*yh).sum(axis=0)/regional_load.sum(axis=0)
    direct = area_direct_cef(Parea.sum(axis=0), Carea.sum(axis=0), aflow.sum(axis=0))
    ds = fine-hourly_month; dt = hourly_month-direct; total=fine-direct
    return ScaleOutputs(fine,hourly_month,direct,ds,dt,total,total-ds-dt,yh,aflow)
