#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="configs/paper_main.toml")
    ap.add_argument("--skip-wind",action="store_true")
    ap.add_argument("--skip-cross-period",action="store_true")
    ap.add_argument("--skip-ac",action="store_true")
    a=ap.parse_args(); root=Path(__file__).resolve().parents[1]
    cfg=str(root/a.config)
    subprocess.check_call([sys.executable,"-m","scale_residual.cli","download-data","--config",cfg],cwd=root)
    subprocess.check_call([sys.executable,"-m","scale_residual.cli","natural-month","--config",cfg],cwd=root)
    if not a.skip_cross_period:
        subprocess.check_call([sys.executable,"-m","scale_residual.cli","cross-period","--config",cfg],cwd=root)
    if not a.skip_wind:
        subprocess.check_call([sys.executable,"-m","scale_residual.cli","wind-stress","--config",cfg],cwd=root)
    if not a.skip_ac:
        subprocess.check_call([sys.executable,"-m","scale_residual.cli","ac-audit","--config",cfg],cwd=root)
    subprocess.check_call([sys.executable,"-m","scale_residual.cli","verify","--config",cfg],cwd=root)
if __name__=="__main__": main()
