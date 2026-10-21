import numpy as np, json
from pathlib import Path
from formal_generation.paper_physics_backend import prepared
from state_comparators.state_utils import descriptor,history_baseline_profile,clip_candidate

def test_hidden_state_contract():
 z=np.load(Path(__file__).parents[1]/'artifacts/calibration_120_state.npz'); assert set(z.files)=={'state_profile_week','state_active','state_week_mean_mw'}; assert z['state_profile_week'].shape==(120,168,48); assert np.isfinite(z['state_profile_week']).all()
def test_no_formal_reference_in_hidden_state_metadata():
 d=json.loads((Path(__file__).parents[1]/'artifacts/calibration_120_state.json').read_text()); assert d['contains_formal_target'] is False
def test_profiles_are_stable():
 z=np.load(Path(__file__).parents[1]/'artifacts/calibration_120_state.npz'); s=z['state_profile_week']; assert s.min()>=0; assert s[:,:,3:].max()<=1.00001; assert s[:,:,:3].max()<3
def test_descriptor_dimension():
 root=Path(__file__).parents[1]; cal=np.load(root/'formal_generation/artifacts/calibration_120.npz'); d=descriptor(cal['x_month'][:2],cal['x_history'][:2]); assert d.shape==(2,424)
def test_baseline_profile():
 root=Path(__file__).parents[1]; cal=np.load(root/'formal_generation/artifacts/calibration_120.npz'); *_,gen=prepared(); b=history_baseline_profile(cal['x_month'][:2],cal['x_history'][:2],gen); assert b.shape==(2,168,48); assert np.isfinite(b).all()
def test_clip_candidate():
 x=np.random.default_rng(1).normal(size=(168,25)); a=np.ones(25,bool); y=clip_candidate(x,a); assert np.all(y>=0); assert y[:,3:].max()<=1;assert y[:,:3].max()<=3
