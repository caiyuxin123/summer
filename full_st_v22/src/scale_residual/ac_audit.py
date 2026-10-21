from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def _ac_branch_state(bus: pd.DataFrame, branch: pd.DataFrame, base_mva: float = 100.0):
    """Build an internally balanced lossy AC audit state from RTS-GMLC data.

    The published bus voltage magnitudes/angles and branch pi-model parameters are
    used to compute complex terminal powers. A deterministic local source/load pair
    is then constructed so that nodal active-power balance is exact for that AC
    branch state. This is an identity/loss-term audit, not an OPF dispatch case.
    """
    bids=bus["Bus ID"].astype(int).to_numpy(); bi={b:i for i,b in enumerate(bids)}; n=len(bids)
    V=bus["V Mag"].to_numpy(float)*np.exp(1j*np.deg2rad(bus["V Angle"].to_numpy(float)))
    Q=np.zeros((n,n),float); d=np.zeros(n,float); terminal=np.zeros(n,float)
    for _,r in branch.iterrows():
        i=bi[int(r["From Bus"])]; j=bi[int(r["To Bus"])]
        rr=float(r["R"]); xx=float(r["X"]); bb=float(r["B"])
        z=complex(rr,xx)
        if abs(z)==0:  # no such line in the bundled RTS data, kept for robustness
            continue
        y=1/z
        tap=float(r["Tr Ratio"]); tap=1.0 if abs(tap)<1e-15 else tap
        Yff=(y+1j*bb/2)/(tap*tap); Yft=-y/tap
        Ytf=-y/tap; Ytt=y+1j*bb/2
        If=Yff*V[i]+Yft*V[j]; It=Ytf*V[i]+Ytt*V[j]
        pf=float((base_mva*V[i]*np.conj(If)).real)
        pt=float((base_mva*V[j]*np.conj(It)).real)
        terminal[i]+=pf; terminal[j]+=pt
        # Decompose terminal powers into nonnegative midpoint transfer plus losses.
        if pf>=0 and pt<=0:
            loss=max(pf+pt,0.0); q=(pf-pt)/2
            Q[i,j]+=q; d[i]+=loss/2; d[j]+=loss/2
        elif pt>=0 and pf<=0:
            loss=max(pf+pt,0.0); q=(pt-pf)/2
            Q[j,i]+=q; d[i]+=loss/2; d[j]+=loss/2
        elif pf>=0 and pt>=0:
            # Both terminals feed a lossy/shunt branch: no through-transfer is needed.
            d[i]+=pf; d[j]+=pt
        else:
            # Active-power generation by a passive branch would violate the audit
            # assumptions. The bundled RTS snapshot does not trigger this case.
            raise RuntimeError(f"non-passive branch active-power pattern on {r['UID']}: {pf}, {pt}")

    # Add the same positive base at both source and load sides; net injection is
    # exactly the branch terminal injection while all regions have positive load.
    common=5.0+0.05*np.maximum(bus["MW Load"].to_numpy(float),0.0)
    local_gen=np.maximum(terminal,0.0)+common
    load=np.maximum(-terminal,0.0)+common
    # Exact consistency check against transfer/loss representation.
    bal=local_gen+Q.sum(axis=0)-load-d-Q.sum(axis=1)
    if np.max(np.abs(bal))>1e-7:
        raise RuntimeError(f"AC audit power balance did not close: {np.max(np.abs(bal))}")
    return bids,Q,d,local_gen,load,bal


def audit_rts_ac(raw_dir: str | Path) -> dict:
    raw_dir=Path(raw_dir)
    bus=pd.read_csv(raw_dir/"bus.csv"); branch=pd.read_csv(raw_dir/"branch.csv")
    bids,Q,d,local_gen,load,bal=_ac_branch_state(bus,branch)
    n=len(bids); areas=sorted(bus["Area"].astype(int).unique()); C=len(areas)
    area_of=dict(zip(bids,bus["Area"].astype(int)))
    ai={a:k for k,a in enumerate(areas)}

    # Deterministic heterogeneous operational-carbon labels for identity auditing.
    # They are deliberately not interpreted as a real dispatch emissions scenario.
    base_ef={areas[0]:820.0,areas[1]:490.0,areas[2]:50.0}
    ef=np.array([base_ef[area_of[int(b)]]*(0.92+0.16*((int(b)*37)%101)/100.0) for b in bids])
    carbon=local_gen*ef

    A=np.diag(load+d+Q.sum(axis=1))-Q.T
    x=np.linalg.solve(A,carbon)

    L=np.zeros(C); D=np.zeros(C); B=np.zeros(C); a=np.zeros(C)
    for i,b in enumerate(bids):
        c=ai[area_of[int(b)]]; L[c]+=load[i]; D[c]+=d[i]; B[c]+=carbon[i]
    for c in range(C):
        ids=[i for i,b in enumerate(bids) if ai[area_of[int(b)]]==c]
        a[c]=np.dot(load[ids],x[ids])/L[c]

    F=np.zeros((C,C))
    for i in range(n):
        ci=ai[area_of[int(bids[i])]]
        for j in np.where(Q[i]>0)[0]:
            cj=ai[area_of[int(bids[j])]]
            if ci!=cj: F[ci,cj]+=Q[i,j]
    AC=np.diag(L+D+F.sum(axis=1))-F.T
    y=np.linalg.solve(AC,B); e=a-y

    rbd=np.zeros(C); rloss=np.zeros(C)
    for i in range(n):
        ci=ai[area_of[int(bids[i])]]
        for j in np.where(Q[i]>0)[0]:
            cj=ai[area_of[int(bids[j])]]
            if ci==cj: continue
            v=Q[i,j]*(x[i]-a[ci]); rbd[cj]+=v; rbd[ci]-=v
        rloss[ci]+=d[i]*(a[ci]-x[i])
    lhs=AC@e; rhs=rbd+rloss; closure=lhs-rhs
    decomp_direct=(lhs-rbd-rloss)
    b_l1=float(np.sum(np.abs(rbd))); l_l1=float(np.sum(np.abs(rloss)))
    return {
        "case":"RTS-GMLC-73 AC audit snapshot","n_bus":n,"n_branch":len(branch),"n_regions":C,
        "max_power_balance_mw":float(np.max(np.abs(bal))),
        "total_active_loss_mw":float(np.sum(d)),
        "max_identity_closure":float(np.max(np.abs(closure))),
        "max_boundary_loss_decomposition_closure":float(np.max(np.abs(decomp_direct))),
        "boundary_exposure_l1":b_l1,"loss_exposure_l1":l_l1,
        "boundary_share_pct":float(100*b_l1/(b_l1+l_l1+1e-30)),
        "spatial_residual_mae":float(np.mean(np.abs(e))),
    }


def run_ac_audit(raw_dir: str | Path) -> pd.DataFrame:
    return pd.DataFrame([audit_rts_ac(raw_dir)])
