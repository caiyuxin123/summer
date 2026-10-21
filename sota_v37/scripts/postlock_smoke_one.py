import argparse,json,time,numpy as np
from pathlib import Path
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
from formal_generation.paper_physics_backend import prepared
from state_comparators.selection import fit_model
from state_comparators.common_projection import project_candidate
ap=argparse.ArgumentParser();ap.add_argument('--method',required=True);a=ap.parse_args();root=Path(__file__).resolve().parents[1];cal=np.load(root/'formal_generation/artifacts/calibration_120.npz');st=np.load(root/'artifacts/calibration_120_state.npz');lock=json.loads((root/'artifacts/selection_v36/model_lock_v36.json').read_text());tr=np.array(lock['train_indices']);i=int(lock['validation_indices'][0]);*_,net,gen=prepared();p=lock['selected'][a.method]['params'];t=time.time();m=fit_model(a.method,p,cal['x_month'][tr],cal['x_history'][tr],st['state_profile_week'][tr],cal['graph'],gen,lock['selection_seed']);pred=m.predict(cal['x_month'][[i]],cal['x_history'][[i]],st['state_active'][[i]])[0];r=project_candidate(pred,cal['x_month'][i],net,gen);err=(r['fine_month']-cal['y'][i])*1000;d={'method':a.method,'scenario_index':i,'MAE_kg_per_MWh':float(np.mean(np.abs(err))),'RMSE_kg_per_MWh':float(np.sqrt(np.mean(err**2))),'MaxAE_kg_per_MWh':float(np.max(np.abs(err))),'runtime_fit_plus_projection_s':float(time.time()-t),'audit':r['audit'],'used_for_selection':False,'formal_target_used':False,'formal_reference_used':False};pout=root/f'artifacts/selection_v36/smoke_{a.method}.json';pout.write_text(json.dumps(d,indent=2));print(json.dumps(d,indent=2))
