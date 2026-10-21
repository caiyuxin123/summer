from __future__ import annotations
import json,itertools,hashlib,time
from pathlib import Path
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
from formal_generation.paper_physics_backend import prepared
from .linear_models import PCAStateForecast,FourierDSE
from .pi_gnn import PIGNNState

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def split(n,seed):
 rng=np.random.default_rng(seed); q=rng.permutation(n); return q[:96],q[96:]
def metrics(y,p,active=None):
 e=np.asarray(p,float)-np.asarray(y,float)
 return {'MAE':float(np.mean(np.abs(e))),'RMSE':float(np.sqrt(np.mean(e**2))),'MaxAE':float(np.max(np.abs(e)))}
def params_grid(spec):
 ks=list(spec); return [dict(zip(ks,v)) for v in itertools.product(*[spec[k] for k in ks])]
def fit_model(key,params,xm,xh,y,graph,gen,seed):
 if key=='sf_star': return PCAStateForecast(params['pca_components'],params['ridge_alpha'],False).fit(xm,xh,y,gen)
 if key=='pm_star': return PCAStateForecast(params['pca_components'],params['ridge_alpha'],True).fit(xm,xh,y,gen)
 if key=='pm_dse': return FourierDSE(params['harmonics'],params['ridge_alpha']).fit(xm,xh,y,gen)
 if key=='pi_gnn': return PIGNNState(params['hidden'],params['lr'],params['epochs'],params['weight_decay'],seed).fit(xm,xh,y,graph,gen)
 raise KeyError(key)
def run(calibration,state_npz,grid_path,outdir,protocol_path=None,projection_path=None):
 cal=np.load(calibration); st=np.load(state_npz); grid=json.loads(Path(grid_path).read_text()); y=st['state_profile_week']; active=st['state_active']; xm=cal['x_month'];xh=cal['x_history']; graph=cal['graph']; _,_,_,gen=prepared(); tr,va=split(len(y),grid['split_seed']); out=Path(outdir);out.mkdir(parents=True,exist_ok=True)
 summaries={}
 for key in ['sf_star','pm_star','pm_dse','pi_gnn']:
  rec=[]
  for i,p in enumerate(params_grid(grid[key])):
   t=time.time(); m=fit_model(key,p,xm[tr],xh[tr],y[tr],graph,gen,grid['selection_seed']); pred=m.predict(xm[va],xh[va],None); met=metrics(y[va],pred,None); rec.append({'candidate_index':i,'params':p,'metrics':met,'complexity':int(m.complexity),'runtime_s':time.time()-t})
  best=min(rec,key=lambda r:(r['metrics']['MAE'],r['metrics']['RMSE'],r['complexity'],r['candidate_index'])); summaries[key]={'selected':best,'all_candidates':rec}
  (out/f'{key}_selection.json').write_text(json.dumps(summaries[key],indent=2),encoding='utf-8')
 lock={'status':'v3.6.1 state-comparator hyperparameters locked before formal target/reference evaluation','calibration_sha256':sha(calibration),'hidden_state_sha256':sha(state_npz),'grid_sha256':sha(grid_path),'protocol_sha256':sha(protocol_path) if protocol_path else None,'common_projection_sha256':sha(projection_path) if projection_path else None,'split_seed':grid['split_seed'],'selection_seed':grid['selection_seed'],'train_indices':tr.tolist(),'validation_indices':va.tolist(),'selected':{k:v['selected'] for k,v in summaries.items()},'selection_metric':'hidden_state_shape_MAE','formal_reference_used':False,'formal_target_used':False}
 (out/'model_lock_v36.json').write_text(json.dumps(lock,indent=2),encoding='utf-8'); return lock
