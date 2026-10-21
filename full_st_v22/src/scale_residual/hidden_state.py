from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog
from .network import DCNetwork, generator_incidence, regional_to_nodal_load

@dataclass
class ProjectionAudit:
    runtime_s: float
    objective: float
    hour_balance_max_mw: float
    monthly_category_max_mwh: float
    max_line_ratio: float


def project_generation(net: DCNetwork, gen: pd.DataFrame,
                       regional_load: np.ndarray, Pprior: np.ndarray,
                       categories, method: str="highs") -> tuple[np.ndarray,np.ndarray,ProjectionAudit]:
    """Physics-consistent projection used in the paper-compatible implementation.

    Regional load profile is fixed by the selected information mode; generator-hour
    states are projected to monthly category totals subject to hourly balance,
    generator capacity and DC line limits, minimizing normalized L1 deviation from
    the independent historical/static prior.
    """
    T,C=regional_load.shape; G=len(gen); L=len(net.rating); N=T*G
    Lbus=regional_to_nodal_load(net, regional_load)
    Cg=generator_incidence(net,gen); Hg=net.H@Cg

    # Equalities: hourly power balance + all but one monthly category equation.
    Abal=sparse.kron(sparse.eye(T,format="csr"), np.ones((1,G)), format="csr")
    bbal=regional_load.sum(axis=1)
    cat_rows=[]; cat_rhs=[]
    for cat in categories[:-1]:  # one equation is redundant with hourly balances
        row=sparse.lil_matrix((1,N))
        ids=cat["ids"]
        for t in range(T): row[0,t*G+ids]=1.0
        cat_rows.append(row.tocsr()); cat_rhs.append(cat["energy"])
    Acat=sparse.vstack(cat_rows,format="csr") if cat_rows else sparse.csr_matrix((0,N))
    Aeq_p=sparse.vstack([Abal,Acat],format="csr")
    beq=np.r_[bbal,np.asarray(cat_rhs,float)]

    # Line constraints H_g p - H load within +/- rating.
    Aflow=sparse.kron(sparse.eye(T,format="csr"), sparse.csr_matrix(Hg), format="csr")
    hload=(net.H@Lbus.T).T
    bup=(net.rating[None,:]+hload).reshape(-1)
    blo=(net.rating[None,:]-hload).reshape(-1)

    # L1 deviations: p-u <= p0 and -p-u <= -p0.
    I=sparse.eye(N,format="csr")
    Z=sparse.csr_matrix((T*L,N))
    Aflow_full=sparse.hstack([Aflow,Z],format="csr")
    Adev1=sparse.hstack([I,-I],format="csr")
    Adev2=sparse.hstack([-I,-I],format="csr")
    Aub=sparse.vstack([Aflow_full,-Aflow_full,Adev1,Adev2],format="csr")
    p0=Pprior.reshape(-1)
    bub=np.r_[bup,blo,p0,-p0]
    Aeq=sparse.hstack([Aeq_p,sparse.csr_matrix((Aeq_p.shape[0],N))],format="csr")

    pmax=np.tile(gen.pmax.to_numpy(float),T)
    bounds=list(zip(np.zeros(N),pmax))+[(0,None)]*N
    weights=np.tile(1.0/np.maximum(gen.pmax.to_numpy(float),1.0),T)
    c=np.r_[np.zeros(N),weights]
    tic=time.perf_counter()
    res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method=method)
    runtime=time.perf_counter()-tic
    if not res.success:
        raise RuntimeError(f"Hidden-state LP failed: {res.message}")
    P=res.x[:N].reshape(T,G)
    flow=P@Hg.T-hload
    cat_err=[P[:,cat["ids"]].sum()-cat["energy"] for cat in categories]
    audit=ProjectionAudit(
        runtime_s=runtime,
        objective=float(res.fun),
        hour_balance_max_mw=float(np.max(np.abs(P.sum(axis=1)-bbal))),
        monthly_category_max_mwh=float(np.max(np.abs(cat_err))),
        max_line_ratio=float(np.max(np.abs(flow)/net.rating[None,:])),
    )
    return P,Lbus,audit
