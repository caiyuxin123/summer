from pathlib import Path
import json,hashlib,numpy as np,statistics,time
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
h=json.loads((ROOT/'protocol/FORMAL_FEB_FROZEN_HASHES_v3.7.json').read_text()); refp=ROOT/'formal_eval/input/reference_only.npz'; targetp=ROOT/'formal_eval/input/target_input_only.npz'; freeze=ROOT/'formal_eval/PHASE_A_BLIND_FREEZE_MANIFEST.json'
if sha(refp)!=h['reference_only_sha256']:raise RuntimeError('reference hash mismatch')
if sha(targetp)!=h['target_input_only_sha256']:raise RuntimeError('target hash mismatch')
ref=np.load(refp); target=np.load(targetp)
if set(ref.files)!=set(h['reference_only_keys']):raise RuntimeError('reference key mismatch')
y=np.asarray(ref['y'][0],float); m=json.loads(freeze.read_text())
if m['status']!='PHASE_A_BLIND_PREDICTIONS_FROZEN_BEFORE_REFERENCE_ATTACH' or m['reference_present'] is not False:raise RuntimeError('invalid blind freeze')
rows=[]
for rr in m['runs']:
 p=ROOT/'formal_eval/blind_projected'/f"{rr['run']}_projected.npz"
 if sha(p)!=rr['projected_sha256']:raise RuntimeError('projected hash changed '+rr['run'])
 pred=np.load(p)['fine_month'].astype(float); e=(pred-y)*1000.0
 rows.append({'run':rr['run'],'method':rr['run'].split('_seed')[0],'prediction_kg_per_mwh':(pred*1000).tolist(),'error_kg_per_mwh':e.tolist(),'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e*e))),'MaxAE':float(np.max(np.abs(e)))})
# target x_month last 3 are frozen strong Direct-M in t/MWh
direct=np.asarray(target['x_month'][0,31:34],float); ed=(direct-y)*1000
baseline={'Direct-M':{'prediction_kg_per_mwh':(direct*1000).tolist(),'error_kg_per_mwh':ed.tolist(),'MAE':float(np.mean(np.abs(ed))),'RMSE':float(np.sqrt(np.mean(ed*ed))),'MaxAE':float(np.max(np.abs(ed)))},'Reference_kg_per_mwh':(y*1000).tolist()}
summary={}
for method in sorted(set(r['method'] for r in rows)):
 xs=[r for r in rows if r['method']==method]
 d={}
 for metric in ['MAE','RMSE','MaxAE']:
  vals=np.array([r[metric] for r in xs],float); d[metric+'_mean']=float(vals.mean());d[metric+'_std']=float(vals.std(ddof=1)) if len(vals)>1 else 0.0;d[metric+'_min']=float(vals.min());d[metric+'_max']=float(vals.max())
 d['n_runs']=len(xs); summary[method]=d
out={'status':'FORMAL_FEB_PHASE_B_EVALUATED_ONCE_FROM_FROZEN_PHASE_A','reference_sha256':sha(refp),'target_sha256':sha(targetp),'blind_freeze_manifest_sha256':sha(freeze),'baseline':baseline,'runs':rows,'summary':summary,'note':'No hyperparameter, feature, candidate, projection, or seed changes are permitted after this evaluation.'}
(ROOT/'formal_eval/FORMAL_FEB_SOTA_RESULTS_v3.7.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
