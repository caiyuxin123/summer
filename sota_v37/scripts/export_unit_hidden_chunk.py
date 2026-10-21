from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from formal_generation.apply_scenario import perturb_base_load
from formal_generation.paper_physics_backend import _run_state, prepared
from paper_physics.features import area_base_load_mw
ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,required=True);ap.add_argument('--stop',type=int,required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
root=Path(__file__).resolve().parents[1];base=np.load(root/'formal_generation/frozen_design/base_history_jan01_07.npy');rows=json.loads((root/'formal_generation/frozen_design/calibration_120_scenario_manifest.json').read_text());*_,net,gen=prepared();lb=area_base_load_mw(net);pmax=gen.pmax.to_numpy(float);prof=[];act=[];means=[]
for i in range(a.start,a.stop):
 t=perturb_base_load(base,rows[i],'target');_,P,_,_=_run_state(t,need_cef=False);raw=np.c_[t,P];scale=np.r_[lb,pmax];pr=raw/scale[None,:];mu=raw.mean(0);prof.append(pr.astype('float32'));act.append((mu>1e-6).astype('uint8'));means.append(mu.astype('float32'))
np.savez_compressed(a.out,index=np.arange(a.start,a.stop,dtype=np.int32),state_profile_week=np.stack(prof),state_active=np.stack(act),state_week_mean_mw=np.stack(means));print(a.start,a.stop,flush=True)
