from pathlib import Path
import sys,json,time,hashlib,argparse,numpy as np
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from formal_generation.paper_physics_backend import prepared
from state_comparators.common_projection import project_candidate

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
ap=argparse.ArgumentParser();ap.add_argument('--candidate',required=True);ap.add_argument('--outdir',required=True);a=ap.parse_args()
if (ROOT/'formal_eval/input/reference_only.npz').exists(): raise RuntimeError('reference_only must be absent during blind projection')
h=json.loads((ROOT/'protocol/FORMAL_FEB_FROZEN_HASHES_v3.7.json').read_text()); target=ROOT/'formal_eval/input/target_input_only.npz'
if sha(target)!=h['target_input_only_sha256']: raise RuntimeError('target hash mismatch')
t=np.load(target); c=np.load(a.candidate); prof=c['state_profile_week'][0]; *_,net,gen=prepared(); t0=time.perf_counter(); r=project_candidate(prof,t['x_month'][0],net,gen); ps=time.perf_counter()-t0
stem=Path(a.candidate).stem.replace('_candidate',''); out=Path(a.outdir);out.mkdir(parents=True,exist_ok=True); npz=out/f'{stem}_projected.npz'; np.savez_compressed(npz,fine_month=r['fine_month'].astype('float64'),direct_month=r['direct_month'].astype('float64'))
meta={'run':stem,'target_sha256':sha(target),'candidate_sha256':sha(a.candidate),'projection_sha256':sha(ROOT/'state_comparators/common_projection.py'),'projected_sha256':sha(npz),'fine_month_t_per_mwh':r['fine_month'].tolist(),'direct_month_t_per_mwh':r['direct_month'].tolist(),'audit':r['audit'],'project_s':ps,'reference_loaded':False}; (out/f'{stem}_projected.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta))
