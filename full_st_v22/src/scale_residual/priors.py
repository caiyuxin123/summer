from __future__ import annotations
import numpy as np
import pandas as pd


def weekday_hour_map(history: pd.DataFrame, target: pd.DataFrame) -> np.ndarray:
    """Map each target timestamp to the matching weekday-hour row in a 7-day history."""
    hmap: dict[tuple[int,int],int] = {}
    for k,ts in enumerate(history.index):
        hmap[(ts.dayofweek, ts.hour)] = k
    missing = {(ts.dayofweek,ts.hour) for ts in target.index} - set(hmap)
    if missing:
        raise ValueError(f"History does not cover all weekday-hour keys: {sorted(missing)}")
    return np.array([hmap[(ts.dayofweek,ts.hour)] for ts in target.index], dtype=int)


def load_prior(history_regional: np.ndarray, target_monthly_energy: np.ndarray,
               map_idx: np.ndarray, mode: str) -> np.ndarray:
    T = len(map_idx); C = len(target_monthly_energy)
    if mode == "flat":
        return np.tile(target_monthly_energy/T, (T,1))
    if mode != "history":
        raise ValueError(mode)
    q = history_regional[map_idx].astype(float).copy()
    for c in range(C):
        den=q[:,c].sum()
        if den <= 0: raise ValueError("Zero historical area load")
        q[:,c] *= target_monthly_energy[c]/den
    return q


def generation_categories(gen: pd.DataFrame, Ptarget: np.ndarray):
    cats=[]
    for a in sorted(gen.area.astype(int).unique()):
        for typ in sorted(gen.loc[gen.area==a,"typ"].unique()):
            ids=np.where((gen.area.to_numpy(int)==a)&(gen.typ.to_numpy()==typ))[0]
            cats.append(dict(area=int(a), typ=str(typ), ids=ids, energy=float(Ptarget[:,ids].sum())))
    return cats


def generation_prior(gen: pd.DataFrame, categories, T: int, mode: str,
                     mapped_history_P: np.ndarray | None = None) -> np.ndarray:
    G=len(gen); out=np.zeros((T,G)); pmax=gen.pmax.to_numpy(float)
    for cat in categories:
        ids=cat["ids"]; E=cat["energy"]
        cap=pmax[ids]; share=cap/cap.sum()
        if mode == "flat":
            out[:,ids]=(E/T)*share[None,:]
        elif mode == "history":
            if mapped_history_P is None: raise ValueError("history generation prior needs mapped_history_P")
            q=mapped_history_P[:,ids].copy(); den=q.sum()
            if E <= 1e-12:
                q[:]=0
            elif den > 1e-12:
                q *= E/den
            else:
                q=(E/T)*share[None,:]
            out[:,ids]=q
        else:
            raise ValueError(mode)
    return out
