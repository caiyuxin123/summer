from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
from .data import ensure_rts_gmlc, read_regional_timeseries, slice_month, slice_dates
from .generator_profile import build_paper_compat_profile, wind_units, aggregate_bus_type
from .network import build_dc_network, regional_to_nodal_load, generator_incidence
from .dispatch import dc_opf_timeseries, dc_opf_timeseries_variable_pmax
from .carbon import nodal_cef_timeseries
from .aggregate import aggregate_state
from .priors import weekday_hour_map, load_prior, generation_categories, generation_prior
from .hidden_state import project_generation
from .metrics import error_metrics
from .io import mkdir, write_json

METHODS = {
    "Static-S": ("flat","flat"),
    "Load-T": ("history","flat"),
    "Gen-T": ("flat","history"),
    "Full-ST": ("history","history"),
}


def _prepare(cfg, include_wind: bool=False):
    paths=ensure_rts_gmlc(cfg)
    bus=pd.read_csv(paths["bus"]); branch=pd.read_csv(paths["branch"])
    net=build_dc_network(bus,branch)
    units=build_paper_compat_profile()
    if include_wind:
        units=pd.concat([units,wind_units()],ignore_index=True)
    gen=aggregate_bus_type(units,bus)
    return paths,bus,branch,net,gen


def _regional_array(df: pd.DataFrame, areas: list[int]) -> np.ndarray:
    cols=[str(a) for a in areas]
    missing=[c for c in cols if c not in df.columns]
    if missing: raise ValueError(f"Missing regional load columns {missing}")
    return df[cols].to_numpy(float)


def _state_outputs(net,gen,regional,P,Lbus):
    Cg=generator_incidence(net,gen); F=P@(net.H@Cg).T-(net.H@Lbus.T).T
    X=nodal_cef_timeseries(net,gen,P,Lbus,F)
    return aggregate_state(net,gen,regional,Lbus,P,F,X),F,X


def run_natural_month(cfg) -> dict:
    paths,bus,branch,net,gen=_prepare(cfg,include_wind=False)
    outdir=mkdir(cfg.output.dir/"natural_month")
    load_all=read_regional_timeseries(paths["load"])
    target_df=slice_month(load_all,cfg.paper.main_year,cfg.paper.main_month)
    hist_df=slice_dates(load_all,cfg.paper.history_start,cfg.paper.history_end)
    Lreg=_regional_array(target_df,net.areas); Lbus=regional_to_nodal_load(net,Lreg)

    # Reference: target-month nodal-hour state used only for offline benchmark/evaluation.
    tic=time.perf_counter()
    Pref,Fref=dc_opf_timeseries(net,gen,Lbus,method=cfg.solver.method)
    Xref=nodal_cef_timeseries(net,gen,Pref,Lbus,Fref)
    ref=aggregate_state(net,gen,Lreg,Lbus,Pref,Fref,Xref)
    reference_runtime=time.perf_counter()-tic

    # Independent 7-day historical prior.
    Lh=_regional_array(hist_df,net.areas); Lbus_h=regional_to_nodal_load(net,Lh)
    Ph,_=dc_opf_timeseries(net,gen,Lbus_h,method=cfg.solver.method)
    map_idx=weekday_hour_map(hist_df,target_df)
    mapped_Ph=Ph[map_idx]
    target_energy=Lreg.sum(axis=0)
    cats=generation_categories(gen,Pref)

    rows=[]; audits=[]; method_runtimes={}
    for method_name,(lmode,gmode) in METHODS.items():
        Lprior=load_prior(Lh,target_energy,map_idx,lmode)
        Pprior=generation_prior(gen,cats,len(target_df),gmode,mapped_Ph if gmode=="history" else None)
        tic=time.perf_counter()
        Pest,Lbus_est,audit=project_generation(net,gen,Lprior,Pprior,cats,method=cfg.solver.method)
        est,Fest,Xest=_state_outputs(net,gen,Lprior,Pest,Lbus_est)
        method_runtime=time.perf_counter()-tic
        method_runtimes[method_name]=method_runtime
        dhat=est.fine_month-est.direct_month
        corrected=ref.direct_month+dhat
        error=corrected-ref.fine_month
        for j,a in enumerate(net.areas):
            rows.append({
                "method":method_name,"area":a,
                "reference":ref.fine_month[j],"direct":ref.direct_month[j],
                "estimated_fine":est.fine_month[j],"estimated_coarse":est.direct_month[j],
                "true_residual":ref.delta_total[j],"estimated_residual":dhat[j],
                "corrected":corrected[j],"error":error[j],
            })
        ar=asdict(audit); ar.update(method=method_name,total_method_runtime_s=method_runtime,**error_metrics(error)); audits.append(ar)
        if cfg.output.write_hourly:
            np.savez_compressed(outdir/f"state_{method_name}.npz",P=Pest,Lreg=Lprior,Lbus=Lbus_est,F=Fest,X=Xest)

    res=pd.DataFrame(rows); aud=pd.DataFrame(audits)
    direct_error=ref.direct_month-ref.fine_month; direct_mae=float(np.mean(np.abs(direct_error)))
    summ=[]
    for method,g in res.groupby("method",sort=False):
        mm=error_metrics(g.error.to_numpy(float)); mm.update(method=method,Direct_MAE=direct_mae,
            improvement_vs_Direct_pct=100*(direct_mae-mm["MAE"])/direct_mae)
        summ.append(mm)
    summary=pd.DataFrame(summ)
    decomposition=pd.DataFrame({
        "area":net.areas,"YF_reference":ref.fine_month,"YH_hourly_area":ref.hourly_coarse_month,
        "YD_direct_month":ref.direct_month,"Delta_total":ref.delta_total,
        "Delta_spatial":ref.delta_spatial,"Delta_temporal":ref.delta_temporal,"closure":ref.closure,
    })
    res.to_csv(outdir/"natural_month_results.csv",index=False)
    summary.to_csv(outdir/"natural_month_summary.csv",index=False)
    decomposition.to_csv(outdir/"residual_decomposition.csv",index=False)
    aud.to_csv(outdir/"physical_audit.csv",index=False)
    write_json(outdir/"runtime.json",{"reference_runtime_s":reference_runtime,"method_runtime_s":method_runtimes})
    write_json(outdir/"metadata.json",{
        "benchmark":"RTS-GMLC release 0.2","year":cfg.paper.main_year,"month":cfg.paper.main_month,
        "hours":len(target_df),"areas":net.areas,"regional_energy_mwh":target_energy,
        "history_start":cfg.paper.history_start,"history_end":cfg.paper.history_end,
        "co2_scope":cfg.paper.co2_scope,
    })
    if cfg.output.write_hourly:
        np.savez_compressed(outdir/"reference_state.npz",P=Pref,F=Fref,X=Xref,Lreg=Lreg,Lbus=Lbus)
    return {"results":res,"summary":summary,"decomposition":decomposition,"audit":aud,"outdir":outdir}


def _wind_availability(wind_df: pd.DataFrame, gen: pd.DataFrame) -> np.ndarray:
    T=len(wind_df); G=len(gen); pmax=np.tile(gen.pmax.to_numpy(float),(T,1))
    for k,r in gen.iterrows():
        if r.typ != "WIND": continue
        # Each paper-compat wind unit is unique by bus, so the official GEN UID is deterministic.
        col=f"{int(r.bus)}_WIND_1"
        if col not in wind_df.columns:
            raise ValueError(f"Official wind series missing column {col}")
        pmax[:,k]=np.minimum(gen.pmax.iloc[k], wind_df[col].to_numpy(float))
    return pmax


def run_wind_stress(cfg) -> dict:
    """Reproduce the fully specified 168-h Jan wind stress experiment.

    Independent history: 2020-01-01..2020-01-07.
    Target:              2020-01-08..2020-01-14.
    Official RTS-GMLC wind series are used as time-varying upper bounds for the
    four wind units. Regional loads are used without scaling; network topology
    and conventional-unit capacities are unchanged from the paper profile.
    """
    paths,bus,branch,net,gen=_prepare(cfg,include_wind=True)
    outdir=mkdir(cfg.output.dir/"wind_stress")
    loads=read_regional_timeseries(paths["load"]); winds=read_regional_timeseries(paths["wind"])
    history_start,history_end="2020-01-01","2020-01-07"
    target_start,target_end="2020-01-08","2020-01-14"
    hist_load=slice_dates(loads,history_start,history_end)
    target_load=slice_dates(loads,target_start,target_end)
    hist_wind=slice_dates(winds,history_start,history_end)
    target_wind=slice_dates(winds,target_start,target_end)
    Lh=_regional_array(hist_load,net.areas); Lt=_regional_array(target_load,net.areas)
    Lbh=regional_to_nodal_load(net,Lh); Lbt=regional_to_nodal_load(net,Lt)
    pmax_h=_wind_availability(hist_wind,gen); pmax_t=_wind_availability(target_wind,gen)
    tic=time.perf_counter()
    Ph,Fh=dc_opf_timeseries_variable_pmax(net,gen,Lbh,pmax_h,method=cfg.solver.method)
    Pt,Ft=dc_opf_timeseries_variable_pmax(net,gen,Lbt,pmax_t,method=cfg.solver.method)
    Xt=nodal_cef_timeseries(net,gen,Pt,Lbt,Ft); ref=aggregate_state(net,gen,Lt,Lbt,Pt,Ft,Xt)
    reference_runtime=time.perf_counter()-tic
    map_idx=weekday_hour_map(hist_load,target_load); mapped_Ph=Ph[map_idx]
    cats=generation_categories(gen,Pt); energy=Lt.sum(axis=0)
    rows=[]; audits=[]
    for name,(lmode,gmode) in METHODS.items():
        Lprior=load_prior(Lh,energy,map_idx,lmode)
        Pprior=generation_prior(gen,cats,len(Lt),gmode,mapped_Ph if gmode=="history" else None)
        tic=time.perf_counter()
        Pest,Lbest,audit=project_generation(net,gen,Lprior,Pprior,cats,method=cfg.solver.method)
        est,_,_=_state_outputs(net,gen,Lprior,Pest,Lbest)
        total_runtime=time.perf_counter()-tic
        dhat=est.fine_month-est.direct_month; corr=ref.direct_month+dhat; err=corr-ref.fine_month
        true=ref.delta_total
        for j,a in enumerate(net.areas):
            rows.append(dict(method=name,area=a,reference=ref.fine_month[j],direct=ref.direct_month[j],
                             true_residual=true[j],estimated_residual=dhat[j],corrected=corr[j],error=err[j]))
        ar=asdict(audit); ar.update(method=name,total_method_runtime_s=total_runtime,
                                   MAE=float(np.mean(np.abs(err))),
                                   DeltaCorr=float(np.corrcoef(true,dhat)[0,1]),
                                   SignAcc_pct=float(100*np.mean(np.sign(true)==np.sign(dhat))))
        audits.append(ar)
    res=pd.DataFrame(rows); aud=pd.DataFrame(audits)
    summ=[]
    for name,g in res.groupby("method",sort=False):
        true=g.true_residual.to_numpy(float); est=g.estimated_residual.to_numpy(float)
        summ.append(dict(method=name,MAE=float(np.mean(np.abs(g.error))),
                         DeltaCorr=float(np.corrcoef(true,est)[0,1]),
                         SignAcc_pct=float(100*np.mean(np.sign(true)==np.sign(est)))))
    summary=pd.DataFrame(summ)
    decomp=pd.DataFrame({"area":net.areas,"YF_reference":ref.fine_month,"YD_direct_month":ref.direct_month,
                         "Delta_total":ref.delta_total,"Delta_spatial":ref.delta_spatial,
                         "Delta_temporal":ref.delta_temporal,"closure":ref.closure})
    res.to_csv(outdir/"wind_results.csv",index=False); summary.to_csv(outdir/"wind_summary.csv",index=False)
    decomp.to_csv(outdir/"wind_residual_decomposition.csv",index=False); aud.to_csv(outdir/"physical_audit.csv",index=False)
    write_json(outdir/"runtime.json",{"reference_and_history_runtime_s":reference_runtime,
                                      "method_runtime_s":{r["method"]:r["total_method_runtime_s"] for r in audits}})
    write_json(outdir/"metadata.json",{
        "benchmark":"RTS-GMLC release 0.2","hours":len(target_load),"areas":net.areas,
        "history_start":history_start,"history_end":history_end,"target_start":target_start,"target_end":target_end,
        "regional_load_scale":1.0,"wind_treatment":"official hourly series as time-varying pmax",
        "wind_units":["122_WIND_1","303_WIND_1","309_WIND_1","317_WIND_1"],
        "conventional_capacity_scaling":1.0,"network_scaling":1.0,"co2_scope":cfg.paper.co2_scope
    })
    return {"results":res,"summary":summary,"decomposition":decomp,"audit":aud,"outdir":outdir}



def _run_conventional_window(cfg, case_name: str, history_start: str, history_end: str,
                             target_start: str, target_end: str) -> dict:
    """Run one fully specified 168-h conventional cross-period experiment.

    Both history and target use official RTS-GMLC regional load data. The target
    node-hour Reference is generated with the same DC-OPF/CEF chain used by the
    natural-month benchmark. The independent 7-day history supplies only the
    load/generation temporal prior used by the hidden-state projection.
    """
    paths,bus,branch,net,gen=_prepare(cfg,include_wind=False)
    outdir=mkdir(cfg.output.dir/"cross_period"/case_name)
    loads=read_regional_timeseries(paths["load"])
    hist_df=slice_dates(loads,history_start,history_end)
    target_df=slice_dates(loads,target_start,target_end)
    if len(hist_df)!=168 or len(target_df)!=168:
        raise ValueError(f"{case_name}: expected 168 h history and target, got {len(hist_df)} / {len(target_df)}")
    Lh=_regional_array(hist_df,net.areas); Lt=_regional_array(target_df,net.areas)
    Lbh=regional_to_nodal_load(net,Lh); Lbt=regional_to_nodal_load(net,Lt)

    tic=time.perf_counter()
    Ph,_=dc_opf_timeseries(net,gen,Lbh,method=cfg.solver.method)
    Pt,Ft=dc_opf_timeseries(net,gen,Lbt,method=cfg.solver.method)
    Xt=nodal_cef_timeseries(net,gen,Pt,Lbt,Ft)
    ref=aggregate_state(net,gen,Lt,Lbt,Pt,Ft,Xt)
    reference_runtime=time.perf_counter()-tic

    map_idx=weekday_hour_map(hist_df,target_df); mapped_Ph=Ph[map_idx]
    cats=generation_categories(gen,Pt); energy=Lt.sum(axis=0)
    rows=[]; audits=[]; method_runtimes={}
    for name,(lmode,gmode) in METHODS.items():
        Lprior=load_prior(Lh,energy,map_idx,lmode)
        Pprior=generation_prior(gen,cats,len(Lt),gmode,mapped_Ph if gmode=="history" else None)
        tic=time.perf_counter()
        Pest,Lbest,audit=project_generation(net,gen,Lprior,Pprior,cats,method=cfg.solver.method)
        est,_,_=_state_outputs(net,gen,Lprior,Pest,Lbest)
        rt=time.perf_counter()-tic; method_runtimes[name]=rt
        dhat=est.fine_month-est.direct_month
        corrected=ref.direct_month+dhat; err=corrected-ref.fine_month
        for j,a in enumerate(net.areas):
            rows.append(dict(case=case_name,method=name,area=a,
                             reference=ref.fine_month[j],direct=ref.direct_month[j],
                             true_residual=ref.delta_total[j],estimated_residual=dhat[j],
                             corrected=corrected[j],error=err[j]))
        ar=asdict(audit); ar.update(case=case_name,method=name,total_method_runtime_s=rt,**error_metrics(err)); audits.append(ar)

    res=pd.DataFrame(rows); aud=pd.DataFrame(audits)
    direct_mae=float(np.mean(np.abs(ref.direct_month-ref.fine_month)))
    summ=[]
    for name,g in res.groupby("method",sort=False):
        mm=error_metrics(g.error.to_numpy(float)); true=g.true_residual.to_numpy(float); est=g.estimated_residual.to_numpy(float)
        mm.update(case=case_name,method=name,Direct_MAE=direct_mae,
                  improvement_vs_Direct_pct=100*(direct_mae-mm["MAE"])/direct_mae,
                  DeltaCorr=float(np.corrcoef(true,est)[0,1]),
                  SignAcc_pct=float(100*np.mean(np.sign(true)==np.sign(est))))
        summ.append(mm)
    summary=pd.DataFrame(summ)
    decomp=pd.DataFrame({"case":case_name,"area":net.areas,"YF_reference":ref.fine_month,
                         "YD_direct_month":ref.direct_month,"Delta_total":ref.delta_total,
                         "Delta_spatial":ref.delta_spatial,"Delta_temporal":ref.delta_temporal,
                         "closure":ref.closure})
    res.to_csv(outdir/"results.csv",index=False); summary.to_csv(outdir/"summary.csv",index=False)
    decomp.to_csv(outdir/"residual_decomposition.csv",index=False); aud.to_csv(outdir/"physical_audit.csv",index=False)
    write_json(outdir/"runtime.json",{"reference_and_history_runtime_s":reference_runtime,"method_runtime_s":method_runtimes})
    write_json(outdir/"metadata.json",{"benchmark":"RTS-GMLC release 0.2","case":case_name,
              "hours":168,"areas":net.areas,"history_start":history_start,"history_end":history_end,
              "target_start":target_start,"target_end":target_end,"wind":"disabled",
              "co2_scope":cfg.paper.co2_scope})
    return {"results":res,"summary":summary,"decomposition":decomp,"audit":aud,"outdir":outdir}


def run_cross_period(cfg) -> dict:
    """Run the current reproducible cross-period conventional validation.

    Both targets use the same frozen independent history 2020-01-01..07.
    Jan target:  2020-01-08..14.
    June target: 2020-06-08..14.
    Using one fixed history prior makes the June window a stricter temporal-transfer
    check and avoids changing the prior between evaluation periods.
    """
    specs={
        "Jan-target":("2020-01-01","2020-01-07","2020-01-08","2020-01-14"),
        "Jun-target":("2020-01-01","2020-01-07","2020-06-08","2020-06-14"),
    }
    out={name:_run_conventional_window(cfg,name,*dates) for name,dates in specs.items()}
    # convenient combined tables
    root=mkdir(cfg.output.dir/"cross_period")
    pd.concat([v["results"] for v in out.values()],ignore_index=True).to_csv(root/"cross_period_results.csv",index=False)
    pd.concat([v["summary"] for v in out.values()],ignore_index=True).to_csv(root/"cross_period_summary.csv",index=False)
    pd.concat([v["decomposition"] for v in out.values()],ignore_index=True).to_csv(root/"cross_period_decomposition.csv",index=False)
    return out


def run_all(cfg, include_wind: bool=True, include_cross_period: bool=True) -> dict:
    out={"natural_month":run_natural_month(cfg)}
    if include_cross_period: out["cross_period"]=run_cross_period(cfg)
    if include_wind: out["wind_stress"]=run_wind_stress(cfg)
    return out
