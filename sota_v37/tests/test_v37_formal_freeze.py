import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def test_v37_core_freeze_and_formal_hashes():
 f=json.loads((ROOT/'formal_eval/V37_FORMAL_FREEZE.json').read_text())
 for p,h in f['files'].items(): assert sha(ROOT/p)==h
 assert sha(ROOT/'formal_eval/input/target_input_only.npz')==f['target_sha256']
 assert sha(ROOT/'formal_eval/input/reference_only.npz')==f['reference_sha256']
def test_phase_a_precedes_reference_evaluation_contract():
 m=json.loads((ROOT/'formal_eval/PHASE_A_BLIND_FREEZE_MANIFEST.json').read_text()); assert m['reference_present'] is False; assert m['candidate_count']==13 and m['projected_count']==13
 r=json.loads((ROOT/'formal_eval/FORMAL_FEB_SOTA_RESULTS_v3.7.json').read_text()); assert r['blind_freeze_manifest_sha256']==sha(ROOT/'formal_eval/PHASE_A_BLIND_FREEZE_MANIFEST.json')
def test_formal_results_are_complete():
 r=json.loads((ROOT/'formal_eval/FORMAL_FEB_SOTA_RESULTS_v3.7.json').read_text()); assert len(r['runs'])==13; assert r['summary']['pi_gnn']['n_runs']==10; assert all(r['summary'][k]['n_runs']==1 for k in ['sf_star','pm_star','pm_dse'])
