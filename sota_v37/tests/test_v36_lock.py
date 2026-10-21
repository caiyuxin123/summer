import hashlib,json
from pathlib import Path
import numpy as np
from formal_generation.paper_physics_backend import prepared
from state_comparators.common_projection import project_candidate

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_no_formal_feb_payload_in_v36_package():
 root=Path(__file__).parents[1]; assert not (root/'formal_generation/artifacts/formal_feb').exists()
def test_lock_is_preformal_and_complete():
 root=Path(__file__).parents[1]; d=json.loads((root/'artifacts/selection_v36/model_lock_v36.json').read_text()); assert d['formal_reference_used'] is False and d['formal_target_used'] is False; assert set(d['selected'])=={'sf_star','pm_star','pm_dse','pi_gnn'}; assert d['hidden_state_sha256']==sha(root/'artifacts/calibration_120_state.npz'); assert d['calibration_sha256']==sha(root/'formal_generation/artifacts/calibration_120.npz'); assert d['grid_sha256']==sha(root/'protocol/STATE_MODEL_GRID_v3.6.json'); assert d['protocol_sha256']==sha(root/'protocol/PAPER_ALIGNED_COMPARATOR_PROTOCOL_v3.6.json'); assert d['common_projection_sha256']==sha(root/'state_comparators/common_projection.py')
def test_oracle_state_common_projection_is_lossless_at_paper_scale():
 root=Path(__file__).parents[1]; cal=np.load(root/'formal_generation/artifacts/calibration_120.npz'); st=np.load(root/'artifacts/calibration_120_state.npz'); lock=json.loads((root/'artifacts/selection_v36/model_lock_v36.json').read_text()); i=int(lock['validation_indices'][0]); *_,net,gen=prepared(); r=project_candidate(st['state_profile_week'][i],cal['x_month'][i],net,gen); mae=float(np.mean(np.abs(r['fine_month']-cal['y'][i]))*1000); assert mae<1e-3; assert r['audit']['max_line_ratio']<=1.0+1e-8; assert r['audit']['max_load_month_err_mwh']<=0.200001; assert r['audit']['max_gen_month_err_mwh']<=0.200001; assert r['audit']['max_directed_flow_err_mwh']<=0.200001

def test_paired_seed_freeze():
 root=Path(__file__).parents[1]; p=json.loads((root/'protocol/PAPER_ALIGNED_COMPARATOR_PROTOCOL_v3.6.json').read_text()); assert p['fairness']['paired_model_seeds']==[20260929+i for i in range(10)]

def test_selection_does_not_use_hidden_activity_mask():
 root=Path(__file__).parents[1]; src=(root/'state_comparators/selection.py').read_text(); assert 'predict(xm[va],xh[va],active[va])' not in src; assert 'mask.repeat' not in src
