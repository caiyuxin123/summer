#!/usr/bin/env python3
"""Read-only verification of the frozen v2.2 + v3.7 artifacts.
The optional --tests mode executes the original test suites in separate CWDs.
This script DOES NOT rerun/retrain models or edit any frozen inputs, predictions or outputs.
"""
import argparse,hashlib,json,subprocess,sys,os
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; MAIN=ROOT/'full_st_v22';SOTA=ROOT/'sota_v37'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--tests',action='store_true'); a=ap.parse_args()
 checks=[]
 def check(name,ok,detail=None):
  checks.append({'name':name,'passed':bool(ok),'detail':detail})
 freeze=json.loads((SOTA/'formal_eval/V37_FORMAL_FREEZE.json').read_text())
 for rel,h in freeze['files'].items():check('v3.7 freeze: '+rel,sha(SOTA/rel)==h)
 for rel,h in [('formal_eval/input/target_input_only.npz',freeze['target_sha256']),('formal_eval/input/reference_only.npz',freeze['reference_sha256'])]:check('v3.7 '+rel,sha(SOTA/rel)==h)
 result=json.loads((SOTA/'formal_eval/FORMAL_FEB_SOTA_RESULTS_v3.7.json').read_text())
 fm=json.loads((SOTA/'formal_eval/PHASE_A_BLIND_FREEZE_MANIFEST.json').read_text())
 check('blind phase A: frozen without Reference',fm['reference_present'] is False and fm['candidate_count']==13 and fm['projected_count']==13)
 for rr in fm['runs']:
  p=SOTA/'formal_eval/blind_projected'/f"{rr['run']}_projected.npz"
  check('frozen projected '+rr['run'],p.exists() and sha(p)==rr['projected_sha256'])
 target=np.load(SOTA/'formal_eval/input/target_input_only.npz',allow_pickle=False)
 ref=np.load(SOTA/'formal_eval/input/reference_only.npz',allow_pickle=False)
 check('blind target input exactly x_month/x_history/graph',set(target.files)=={'x_month','x_history','graph'})
 check('Reference file only contains y',set(ref.files)=={'y'})
 orig=pd.read_csv(MAIN/'outputs/paper_main/natural_month/residual_decomposition.csv').sort_values('area')
 difref=np.max(np.abs(ref['y'][0]*1000-orig.YF_reference.to_numpy()))
 difdirect=np.max(np.abs(target['x_month'][0,31:34]*1000-orig.YD_direct_month.to_numpy()))
 check('v2.2 Reference matches v3.7 [kgCO2/MWh] within 1e-3',difref<1e-3,float(difref))
 check('v2.2 Direct-M matches v3.7 [kgCO2/MWh] within 1e-3',difdirect<1e-3,float(difdirect))
 for name in ('generator_profile.py','network.py','carbon.py','aggregate.py','dispatch.py'):
  p1=MAIN/'src/scale_residual'/name;p2=SOTA/'paper_physics'/name
  check('shared physics byte equality: '+name,sha(p1)==sha(p2))
 frommain=pd.read_csv(MAIN/'outputs/paper_main/natural_month/natural_month_summary.csv')
 full=float(frommain.loc[frommain.method=='Full-ST','MAE'].iloc[0]);direct=float(frommain.loc[frommain.method=='Full-ST','Direct_MAE'].iloc[0]);
 check('Full-ST v2.2 frozen MAE',abs(full-1.1170817273901814)<1e-8,full)
 check('Direct-M v2.2 frozen MAE',abs(direct-140.1596268)<1e-5,direct)
 exp=(MAIN/'reference/expected_natural_month_summary.csv'); expected=pd.read_csv(exp)
 for _,r in expected.iterrows():
  actual=float(frommain.loc[frommain.method==r.method,'MAE'].iloc[0]);check('v2.2 expected '+r.method,abs(actual-r.MAE)<0.35,actual)
 check('v3.7 13 frozen runs',len(result['runs'])==13)
 check('v3.7 PI-GNN 10 seeds',result['summary']['pi_gnn']['n_runs']==10)
 for method,expected_mae in [('pm_star',0.649),('sf_star',0.769),('pi_gnn',0.990),('pm_dse',1.170)]:
  actual=result['summary'][method]['MAE_mean'];check('v3.7 frozen '+method,abs(actual-expected_mae)<0.002,actual)
 if a.tests:
  for label,folder,pp in [('Full-ST',MAIN,str(MAIN/'src')),('SOTA',SOTA,str(SOTA))]:
   env=dict(os.environ,PYTHONPATH=pp)
   run=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=folder,env=env,capture_output=True,text=True)
   check('pytest '+label,run.returncode==0,(run.stdout+run.stderr)[-800:])
 passed=all(x['passed'] for x in checks)
 print(json.dumps({'passed':passed,'check_count':len(checks),'failed':[x for x in checks if not x['passed']],'checks':checks if not passed else [],'notes':'Frozen-result/checkpoint integrity only; not an independent Feb 696h+120 scenarios end-to-end rerun.'},ensure_ascii=False,indent=2))
 return 0 if passed else 1
if __name__=='__main__':sys.exit(main())
