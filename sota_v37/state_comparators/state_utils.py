from __future__ import annotations
from functools import lru_cache
import numpy as np
from paper_physics.features import category_schema, area_base_load_mw


def cat_meta(gen):
    sch=category_schema(gen)
    return [(a,t,np.asarray(ids,dtype=int),float(cap)) for a,t,ids,cap in sch]


def aggregate_gen_categories(P, gen):
    out=np.zeros((len(P),len(cat_meta(gen))),dtype=float)
    for c,(_a,_t,ids,_cap) in enumerate(cat_meta(gen)): out[:,c]=np.asarray(P)[:,ids].sum(axis=1)
    return out


def clip_candidate(profile, active=None):
    z=np.asarray(profile,float).copy(); z[:,:3]=np.clip(z[:,:3],0.0,3.0); z[:,3:]=np.clip(z[:,3:],0.0,1.0)
    if active is not None: z[:,~np.asarray(active,bool)]=0.0
    return z.astype(np.float32)


def descriptor(x_month,x_history):
    xm=np.asarray(x_month,float); h=np.asarray(x_history,float)
    if h.ndim==3:h=h[None,...]
    if xm.ndim==1:xm=xm[None,:]
    if h.shape[-3:]!=(168,3,5):raise ValueError(h.shape)
    dayhour=h.reshape(len(h),7,24,3,5).mean(axis=1).reshape(len(h),-1); mu=h.mean(axis=1).reshape(len(h),-1); sd=h.std(axis=1).reshape(len(h),-1)
    return np.concatenate([xm,mu,sd,dayhour],axis=1).astype(np.float64)


def _frozen_unit_history(gen):
    # Target-external Jan 1-7 dispatch profile; deterministic and shared by all comparators.
    from formal_generation.paper_physics_backend import load_window,_run_state,feb2020_weekday_hour_indices
    hist=load_window('2020-01-01','2020-01-07'); _lb,P,_F,_=_run_state(hist,need_cef=False)
    prof=P/np.maximum(gen.pmax.to_numpy(float),1e-9)[None,:]
    w=np.bincount(feb2020_weekday_hour_indices(),minlength=168).astype(float)
    return prof,w


def history_baseline_profile(x_month,x_history,gen):
    xm=np.asarray(x_month,float); h=np.asarray(x_history,float)
    if h.ndim==3:h=h[None,...]
    if xm.ndim==1:xm=xm[None,:]
    n=len(h); G=len(gen); out=np.zeros((n,168,3+G),float); out[:,:,:3]=np.maximum(h[:,:,:,0],0.0)
    hp,w=_frozen_unit_history(gen); cats=cat_meta(gen); pmax=gen.pmax.to_numpy(float)
    for i in range(n):
        for c,(_a,_typ,ids,cap) in enumerate(cats):
            target_cf=float(xm[i,3+c]); weighted_energy=float((hp[:,ids]*pmax[ids][None,:]*w[:,None]).sum()); target_energy=target_cf*float(w.sum())*cap
            scale=target_energy/max(weighted_energy,1e-9) if target_energy>0 else 0.0
            out[i][:,3+ids]=np.clip(hp[:,ids]*scale,0.0,1.0)
    return out.astype(np.float32)


def padded_area_target(profile, gen):
    profile=np.asarray(profile,float)
    if profile.ndim==2:profile=profile[None,...]
    areas=gen.area.to_numpy(int); by={a:np.where(areas==a)[0].tolist() for a in (1,2,3)}; K=max(len(v) for v in by.values())
    out=np.zeros((len(profile),168,3,1+K),np.float32);mask=np.zeros((3,1+K),np.float32);mask[:,0]=1
    for a in (1,2,3):
        out[:,:,a-1,0]=profile[:,:,a-1]
        for s,g in enumerate(by[a]):out[:,:,a-1,1+s]=profile[:,:,3+g];mask[a-1,1+s]=1
    return out,mask,by


def padded_to_profile(padded,gen):
    p=np.asarray(padded,float)
    if p.ndim==3:p=p[None,...]
    areas=gen.area.to_numpy(int);by={a:np.where(areas==a)[0].tolist() for a in (1,2,3)}
    out=np.zeros((len(p),168,3+len(gen)),float);out[:,:,:3]=p[:,:,:,0]
    for a in (1,2,3):
        for s,g in enumerate(by[a]):out[:,:,3+g]=p[:,:,a-1,1+s]
    return out.astype(np.float32)
