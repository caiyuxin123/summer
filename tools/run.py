#!/usr/bin/env python3
"""Isolated sub-project runners; default is read-only verification.
No call implicitly refits SOTA models or touches frozen predictions.
"""
import argparse,subprocess,sys,os
from pathlib import Path
R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('task',choices=['verify','verify-tests','full-st-verify','full-st-reproduce','sota-tests','sota-phase-b'])
a=p.parse_args()
def execute(folder,cmd,py=''):
 env=os.environ.copy()
 if py:env['PYTHONPATH']=str(folder/py)
 subprocess.check_call(cmd,cwd=folder,env=env)
if a.task=='verify':execute(R,[sys.executable,'tools/verify_unified.py'])
elif a.task=='verify-tests':execute(R,[sys.executable,'tools/verify_unified.py','--tests'])
elif a.task=='full-st-verify':execute(R/'full_st_v22',[sys.executable,'-m','scale_residual.cli','verify','--config','configs/paper_main.toml'],'src')
elif a.task=='full-st-reproduce':execute(R/'full_st_v22',[sys.executable,'run_reproduce.py'],'src')
elif a.task=='sota-tests':execute(R/'sota_v37',[sys.executable,'-m','pytest','-q'],'.')
elif a.task=='sota-phase-b':execute(R/'sota_v37',[sys.executable,'formal_eval/evaluate_phase_b_once.py'],'.')
