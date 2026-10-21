from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix
from paper_physics.network import generator_incidence,regional_to_nodal_load
from paper_physics.features import category_schema,area_base_load_mw
from paper_physics.carbon import nodal_cef_timeseries
from paper_physics.aggregate import aggregate_state
from formal_generation.paper_physics_backend import feb2020_weekday_hour_indices, load_window, _run_state

MONTH_BAND_MWH=0.2  # only covers float32 serialization mismatch; << paper reporting precision

def _extract_targets(xm,T,gen,net):
    xm=np.asarray(xm,float).reshape(-1); cats=category_schema(gen)
    loadE=xm[:3]*area_base_load_mw(net)*T
    genE=np.array([xm[3+c]*T*cap for c,(_a,_t,_ids,cap) in enumerate(cats)])
    total=loadE.sum(); pairs=[]; vals=[]; k=25
    for a in net.areas:
        for b in net.areas:
            if a!=b: pairs.append((a,b)); vals.append(xm[k]*total); k+=1
    return loadE,genE,pairs,np.asarray(vals,float)

def _expand_prior(profile_week,xm,gen,net):
    idx=feb2020_weekday_hour_indices(); S=np.asarray(profile_week,float)[idx]
    lb=area_base_load_mw(net); caps=np.asarray([x[3] for x in category_schema(gen)],float)
    return np.maximum(S[:,:3],0.0)*lb[None,:], np.maximum(S[:,3:],0.0)*caps[None,:]

def project_candidate(profile_week,xm,net,gen,method='highs'):
    # Solve on the frozen 168 weekday-hour representatives and use occurrence weights
    # (4 or 5 in Feb-2020) for exact monthly constraints. This is algebraically the same
    # calendar representation used by the SOTA input freeze and avoids a 696 h LP blow-up.
    S=np.asarray(profile_week,float)
    if S.shape[0]!=168: raise ValueError(S.shape)
    T=168; A=3; G=len(gen); cats=category_schema(gen); C=len(cats)
    idx=feb2020_weekday_hour_indices(); w=np.bincount(idx,minlength=168).astype(float); Tmonth=int(w.sum())
    loadE,genE,pairs,flowE=_extract_targets(xm,Tmonth,gen,net)
    lb=area_base_load_mw(net); cap=np.asarray([x[3] for x in cats],float)
    priorL=np.maximum(S[:,:3],0.0)*lb[None,:]; priorP=np.maximum(S[:,3:],0.0)*gen.pmax.to_numpy(float)[None,:]
    if priorP.shape!=(T,G): raise ValueError(f'unit-level candidate profile must have 3+{G} channels, got {S.shape}')
    Cg=generator_incidence(net,gen); Hg=net.H@Cg
    N=len(net.bus_ids); Cload=np.zeros((N,A))
    for j,a in enumerate(net.areas):
        ids=np.where(net.area_of_bus==a)[0]; Cload[ids,j]=net.load_share[ids]
    Hl=net.H@Cload
    tie=[]
    for l in range(len(net.rating)):
        af=int(net.area_of_bus[net.f_idx[l]]); at=int(net.area_of_bus[net.t_idx[l]])
        if af!=at: tie.append((l,af,at))
    K=len(tie)
    nP=T*G;nL=T*A;oP=0;oL=nP;oDPp=oL+nL;oDPn=oDPp+T*G;oDLp=oDPn+T*G;oDLn=oDLp+T*A;oQp=oDLn+T*A;oQn=oQp+T*K;n=oQn+T*K
    c=np.zeros(n);lscale=np.maximum(loadE/Tmonth,1.0)
    for t in range(T):
        wt=w[t]
        c[oDPp+t*G:oDPp+(t+1)*G]=wt/(G*np.maximum(gen.pmax.to_numpy(float),1));c[oDPn+t*G:oDPn+(t+1)*G]=c[oDPp+t*G:oDPp+(t+1)*G]
        c[oDLp+t*A:oDLp+(t+1)*A]=wt/(A*lscale);c[oDLn+t*A:oDLn+(t+1)*A]=c[oDLp+t*A:oDLp+(t+1)*A]
        c[oQp+t*K:oQp+(t+1)*K]=wt*1e-8;c[oQn+t*K:oQn+(t+1)*K]=wt*1e-8;c[oP+t*G:oP+(t+1)*G]=wt*1e-10*gen.cost.to_numpy(float)
    er=[];ec=[];ev=[];be=[];row=0
    def add_eq(cols,vals,b):
        nonlocal row;er.extend([row]*len(cols));ec.extend(cols);ev.extend(vals);be.append(float(b));row+=1
    bands=[]
    def add_band(cols,vals,target,tol=MONTH_BAND_MWH): bands.append((cols,vals,float(target),float(tol)))
    for t in range(T):
        for g in range(G): add_eq([oP+t*G+g,oDPp+t*G+g,oDPn+t*G+g],[1,-1,1],priorP[t,g])
        for a in range(A):add_eq([oL+t*A+a,oDLp+t*A+a,oDLn+t*A+a],[1,-1,1],priorL[t,a])
    for cc,(_a,_typ,ids,_cap) in enumerate(cats):
        cols=[];vals=[]
        for t in range(T): cols.extend([oP+t*G+int(g) for g in ids]); vals.extend([w[t]]*len(ids))
        add_band(cols,vals,genE[cc])
    for a in range(A): add_band([oL+t*A+a for t in range(T)],w.tolist(),loadE[a])
    for t in range(T): add_eq([oP+t*G+g for g in range(G)]+[oL+t*A+a for a in range(A)],[1.0]*G+[-1.0]*A,0)
    for t in range(T):
        for kk,(l,_af,_at) in enumerate(tie):
            add_eq([oQp+t*K+kk,oQn+t*K+kk]+[oP+t*G+g for g in range(G)]+[oL+t*A+a for a in range(A)],[1,-1]+(-Hg[l]).tolist()+Hl[l].tolist(),0)
    pidx={p:i for i,p in enumerate(pairs)}
    for p in pairs:
        cols=[];vals=[]
        for t in range(T):
            for kk,(_l,af,at) in enumerate(tie):
                if (af,at)==p: cols.append(oQp+t*K+kk); vals.append(w[t])
                elif (at,af)==p: cols.append(oQn+t*K+kk); vals.append(w[t])
        add_band(cols,vals,flowE[pidx[p]])
    Aeq=coo_matrix((ev,(er,ec)),shape=(row,n)).tocsr();beq=np.asarray(be)
    ir=[];ic=[];iv=[];bu=[];r=0
    def add_ub(cols,vals,b):
        nonlocal r;ir.extend([r]*len(cols));ic.extend(cols);iv.extend(vals);bu.append(float(b));r+=1
    for t in range(T):
        for l in range(len(net.rating)):
            cols=[oP+t*G+g for g in range(G)]+[oL+t*A+a for a in range(A)]
            add_ub(cols,Hg[l].tolist()+(-Hl[l]).tolist(),net.rating[l]);add_ub(cols,(-Hg[l]).tolist()+Hl[l].tolist(),net.rating[l])
    for cols,vals,target,tol in bands:
        add_ub(cols,vals,target+tol);add_ub(cols,[-v for v in vals],-target+tol)
    Aub=coo_matrix((iv,(ir,ic)),shape=(r,n)).tocsr();bub=np.asarray(bu)
    bounds=[];pmax=gen.pmax.to_numpy(float)
    for _t in range(T):bounds.extend([(0,float(x)) for x in pmax])
    bounds.extend([(0,None)]*(n-nP))
    res=linprog(c,A_ub=Aub,b_ub=bub,A_eq=Aeq,b_eq=beq,bounds=bounds,method=method,options={'presolve':True})
    if not res.success:raise RuntimeError('common projection failed: '+res.message)
    Pw=res.x[oP:oL].reshape(T,G);Lw=res.x[oL:oDPp].reshape(T,A);Lbusw=regional_to_nodal_load(net,Lw);Fw=Pw@Hg.T-Lw@Hl.T
    P=Pw[idx];L=Lw[idx];F=Fw[idx];Lbus=regional_to_nodal_load(net,L);X=nodal_cef_timeseries(net,gen,P,Lbus,F);state=aggregate_state(net,gen,L,Lbus,P,F,X)
    catE=np.array([(Pw[:,ids]*w[:,None]).sum() for _a,_t,ids,_cap in cats]);directed=np.zeros(len(pairs))
    for kk,(_l,af,at) in enumerate(tie):
        qpos=res.x[oQp+kk:oQp+T*K:K];qneg=res.x[oQn+kk:oQn+T*K:K];directed[pidx[(af,at)]]+=(qpos*w).sum();directed[pidx[(at,af)]]+=(qneg*w).sum()
    audit={'objective':float(res.fun),'max_load_month_err_mwh':float(np.max(np.abs((Lw*w[:,None]).sum(0)-loadE))),'max_gen_month_err_mwh':float(np.max(np.abs(catE-genE))),'max_directed_flow_err_mwh':float(np.max(np.abs(directed-flowE))),'max_line_ratio':float(np.max(np.abs(Fw)/net.rating[None,:])),'representative_hours':T,'expanded_hours':Tmonth,'monthly_band_mwh':MONTH_BAND_MWH}
    return {'Lreg':L,'P':P,'F':F,'fine_month':state.fine_month/1000.0,'direct_month':state.direct_month/1000.0,'audit':audit}
