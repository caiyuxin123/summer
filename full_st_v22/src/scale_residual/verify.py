from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def verify_outputs(root: Path, output_dir: Path, mae_tol: float=0.35, area_tol: float=0.75) -> dict:
    report={"passed":True,"checks":[]}
    expected=pd.read_csv(root/"reference/expected_natural_month_summary.csv")
    actual=pd.read_csv(output_dir/"natural_month/natural_month_summary.csv")
    for _,r in expected.iterrows():
        a=actual.loc[actual.method==r.method]
        if a.empty:
            report["passed"]=False; report["checks"].append({"name":f"MAE {r.method}","passed":False,"reason":"missing"}); continue
        diff=abs(float(a.MAE.iloc[0])-float(r.MAE))
        ok=diff<=mae_tol; report["passed"] &= ok
        report["checks"].append({"name":f"MAE {r.method}","expected":float(r.MAE),"actual":float(a.MAE.iloc[0]),"abs_diff":diff,"tol":mae_tol,"passed":ok})
    expd=pd.read_csv(root/"reference/expected_natural_month_decomposition.csv")
    actd=pd.read_csv(output_dir/"natural_month/residual_decomposition.csv")
    for _,r in expd.iterrows():
        a=actd.loc[actd.area==r.area].iloc[0]
        for col in ["YF_reference","YD_direct_month","Delta_total","Delta_spatial","Delta_temporal"]:
            diff=abs(float(a[col])-float(r[col])); ok=diff<=area_tol; report["passed"] &= ok
            report["checks"].append({"name":f"Area{int(r.area)} {col}","expected":float(r[col]),"actual":float(a[col]),"abs_diff":diff,"tol":area_tol,"passed":ok})
    # Jan-wind extension: verify both method-level metrics and per-Area residual decomposition.
    wind_dir=output_dir/"wind_stress"
    expw_path=root/"reference/expected_wind_summary.csv"
    if expw_path.exists() and (wind_dir/"wind_summary.csv").exists():
        expw=pd.read_csv(expw_path); actw=pd.read_csv(wind_dir/"wind_summary.csv")
        for _,r in expw.iterrows():
            a=actw.loc[actw.method==r.method]
            if a.empty:
                report["passed"]=False; report["checks"].append({"name":f"Wind MAE {r.method}","passed":False,"reason":"missing"}); continue
            for col,tol in [("MAE",mae_tol),("DeltaCorr",0.03),("SignAcc_pct",0.5)]:
                diff=abs(float(a[col].iloc[0])-float(r[col])); ok=diff<=tol; report["passed"] &= ok
                report["checks"].append({"name":f"Wind {r.method} {col}","expected":float(r[col]),
                                         "actual":float(a[col].iloc[0]),"abs_diff":diff,"tol":tol,"passed":ok})
    expkd_path=root/"reference/expected_wind_key_residuals.csv"
    if expkd_path.exists() and (wind_dir/"wind_residual_decomposition.csv").exists():
        expkd=pd.read_csv(expkd_path); actkd=pd.read_csv(wind_dir/"wind_residual_decomposition.csv")
        for _,r in expkd.iterrows():
            a=actkd.loc[actkd.area==r.area].iloc[0]
            for col in ["YF_reference","YD_direct_month","Delta_total","Delta_spatial","Delta_temporal"]:
                diff=abs(float(a[col])-float(r[col])); ok=diff<=area_tol; report["passed"] &= ok
                report["checks"].append({"name":f"Wind Area{int(r.area)} {col}","expected":float(r[col]),
                                         "actual":float(a[col]),"abs_diff":diff,"tol":area_tol,"passed":ok})

    # Cross-period current reproducible validation.
    cdir=output_dir/"cross_period"
    expc_path=root/"reference/expected_cross_period_summary.csv"
    if expc_path.exists() and (cdir/"cross_period_summary.csv").exists():
        expc=pd.read_csv(expc_path); actc=pd.read_csv(cdir/"cross_period_summary.csv")
        for _,r in expc.iterrows():
            a=actc.loc[(actc.case==r.case)&(actc.method==r.method)]
            if a.empty:
                report["passed"]=False; report["checks"].append({"name":f"Cross {r.case} {r.method}","passed":False,"reason":"missing"}); continue
            for col,tol in [("MAE",mae_tol),("Direct_MAE",mae_tol),("DeltaCorr",0.03),("SignAcc_pct",0.5)]:
                diff=abs(float(a[col].iloc[0])-float(r[col])); ok=diff<=tol; report["passed"] &= ok
                report["checks"].append({"name":f"Cross {r.case} {r.method} {col}","expected":float(r[col]),
                                         "actual":float(a[col].iloc[0]),"abs_diff":diff,"tol":tol,"passed":ok})
    expcd_path=root/"reference/expected_cross_period_decomposition.csv"
    if expcd_path.exists() and (cdir/"cross_period_decomposition.csv").exists():
        expcd=pd.read_csv(expcd_path); actcd=pd.read_csv(cdir/"cross_period_decomposition.csv")
        for _,r in expcd.iterrows():
            a=actcd.loc[(actcd.case==r.case)&(actcd.area==r.area)].iloc[0]
            for col in ["YF_reference","YD_direct_month","Delta_total","Delta_spatial","Delta_temporal"]:
                diff=abs(float(a[col])-float(r[col])); ok=diff<=area_tol; report["passed"] &= ok
                report["checks"].append({"name":f"Cross {r.case} Area{int(r.area)} {col}","expected":float(r[col]),
                                         "actual":float(a[col]),"abs_diff":diff,"tol":area_tol,"passed":ok})

    # Self-contained lossy-AC identity audit.
    aca_dir=output_dir/"ac_audit"
    expa_path=root/"reference/expected_ac_audit.csv"
    if expa_path.exists() and (aca_dir/"ac_identity_audit.csv").exists():
        expa=pd.read_csv(expa_path).iloc[0]; acta=pd.read_csv(aca_dir/"ac_identity_audit.csv").iloc[0]
        for col,tol in [("max_power_balance_mw",1e-8),("total_active_loss_mw",1e-6),
                        ("max_identity_closure",1e-8),("max_boundary_loss_decomposition_closure",1e-8),
                        ("boundary_share_pct",0.05)]:
            diff=abs(float(acta[col])-float(expa[col])); ok=diff<=tol; report["passed"] &= ok
            report["checks"].append({"name":f"AC audit {col}","expected":float(expa[col]),
                                     "actual":float(acta[col]),"abs_diff":diff,"tol":tol,"passed":ok})
    return report
