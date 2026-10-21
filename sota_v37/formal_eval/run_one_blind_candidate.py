from pathlib import Path
import sys, json, time, hashlib, argparse, numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from formal_generation.paper_physics_backend import prepared
from state_comparators.selection import fit_model

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser(); ap.add_argument('--method',required=True,choices=['sf_star','pm_star','pm_dse','pi_gnn']); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--outdir',required=True); a=ap.parse_args()
out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
hashes=json.loads((ROOT/'protocol/FORMAL_FEB_FROZEN_HASHES_v3.7.json').read_text())
target=ROOT/'formal_eval/input/target_input_only.npz'
if sha(target)!=hashes['target_input_only_sha256']: raise RuntimeError('formal target hash mismatch')
if (ROOT/'formal_eval/input/reference_only.npz').exists(): raise RuntimeError('reference_only must be absent during phase A')
z=np.load(target); 
if set(z.files)!=set(hashes['target_input_keys']): raise RuntimeError('formal target key mismatch')
cal=np.load(ROOT/'formal_generation/artifacts/calibration_120.npz'); st=np.load(ROOT/'artifacts/calibration_120_state.npz')
lock=json.loads((ROOT/'artifacts/selection_v36/model_lock_v36.json').read_text()); p=lock['selected'][a.method]['params']; *_,gen=prepared()
t0=time.perf_counter(); model=fit_model(a.method,p,cal['x_month'],cal['x_history'],st['state_profile_week'],cal['graph'],gen,a.seed); train_s=time.perf_counter()-t0
t1=time.perf_counter(); pred=model.predict(z['x_month'],z['x_history'],None); pred_s=time.perf_counter()-t1
if pred.shape!=(1,168,48) or not np.isfinite(pred).all(): raise RuntimeError(pred.shape)
npz=out/f'{a.method}_seed{a.seed}_candidate.npz'; np.savez_compressed(npz,state_profile_week=pred.astype('float32'))
meta={'method':a.method,'seed':a.seed,'params':p,'target_sha256':sha(target),'lock_sha256':sha(ROOT/'artifacts/selection_v36/model_lock_v36.json'),'candidate_sha256':sha(npz),'candidate_shape':list(pred.shape),'candidate_min':float(pred.min()),'candidate_max':float(pred.max()),'complexity':int(model.complexity),'train_s':train_s,'predict_s':pred_s,'reference_loaded':False}
(out/f'{a.method}_seed{a.seed}_candidate.json').write_text(json.dumps(meta,indent=2)+'\n'); print(json.dumps(meta))
