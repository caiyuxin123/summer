from __future__ import annotations
import argparse, json, sys
from .config import load_config
from .data import ensure_rts_gmlc
from .experiments import run_all, run_natural_month, run_wind_stress, run_cross_period
from .plots import plot_natural_month, plot_wind
from .verify import verify_outputs
from .io import write_json


def main(argv=None):
    p=argparse.ArgumentParser(prog="scale-residual")
    sub=p.add_subparsers(dest="cmd",required=True)
    for name in ["download-data","natural-month","cross-period","wind-stress","ac-audit","reproduce","verify"]:
        q=sub.add_parser(name); q.add_argument("--config",default="configs/paper_main.toml")
        if name=="download-data": q.add_argument("--force",action="store_true")
    args=p.parse_args(argv); cfg=load_config(args.config)
    if args.cmd=="download-data":
        paths=ensure_rts_gmlc(cfg,force=args.force)
        for k,v in paths.items(): print(f"{k}: {v}")
        return 0
    if args.cmd=="natural-month":
        out=run_natural_month(cfg)
        if cfg.output.write_plots: plot_natural_month(out["outdir"])
        print(out["summary"].to_string(index=False)); return 0
    if args.cmd=="cross-period":
        out=run_cross_period(cfg)
        for case,v in out.items():
            print(f"\n[{case}]")
            print(v["summary"].to_string(index=False))
        return 0
    if args.cmd=="wind-stress":
        out=run_wind_stress(cfg)
        if cfg.output.write_plots: plot_wind(out["outdir"])
        print(out["summary"].to_string(index=False)); return 0
    if args.cmd=="ac-audit":
        from .ac_audit import run_ac_audit
        df=run_ac_audit(cfg.data.raw_dir); outdir=cfg.output.dir/"ac_audit"; outdir.mkdir(parents=True,exist_ok=True)
        df.to_csv(outdir/"ac_identity_audit.csv",index=False); print(df.to_string(index=False)); return 0
    if args.cmd=="reproduce":
        out=run_all(cfg,include_wind=True,include_cross_period=True)
        if cfg.output.write_plots:
            plot_natural_month(out["natural_month"]["outdir"]); plot_wind(out["wind_stress"]["outdir"])
        from .ac_audit import run_ac_audit
        acdir=cfg.output.dir/"ac_audit"; acdir.mkdir(parents=True,exist_ok=True)
        run_ac_audit(cfg.data.raw_dir).to_csv(acdir/"ac_identity_audit.csv",index=False)
        rep=verify_outputs(cfg.root,cfg.output.dir,cfg.verification.mae_tolerance,cfg.verification.area_value_tolerance)
        write_json(cfg.output.dir/"verification_report.json",rep)
        print(json.dumps(rep,ensure_ascii=False,indent=2)); return 0 if rep["passed"] else 2
    if args.cmd=="verify":
        rep=verify_outputs(cfg.root,cfg.output.dir,cfg.verification.mae_tolerance,cfg.verification.area_value_tolerance)
        print(json.dumps(rep,ensure_ascii=False,indent=2)); return 0 if rep["passed"] else 2
    return 1

if __name__=="__main__": raise SystemExit(main())
