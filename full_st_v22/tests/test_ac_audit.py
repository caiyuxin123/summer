from pathlib import Path
from scale_residual.ac_audit import audit_rts_ac

ROOT=Path(__file__).resolve().parents[1]

def test_self_contained_lossy_ac_identity_audit():
    r=audit_rts_ac(ROOT/'data/raw')
    assert r['n_bus']==73 and r['n_branch']==120 and r['n_regions']==3
    assert r['total_active_loss_mw'] > 0
    assert r['max_power_balance_mw'] < 1e-8
    assert r['max_identity_closure'] < 1e-8
    assert r['max_boundary_loss_decomposition_closure'] < 1e-8
    assert 0 < r['boundary_share_pct'] < 100
