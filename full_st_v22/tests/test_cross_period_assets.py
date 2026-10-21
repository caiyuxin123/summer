from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]

def test_cross_period_expected_assets_are_complete():
    s=pd.read_csv(ROOT/'reference/expected_cross_period_summary.csv')
    assert set(s['case'])=={'Jan-target','Jun-target'}
    assert set(s['method'])=={'Static-S','Load-T','Gen-T','Full-ST'}
    assert len(s)==8
    d=pd.read_csv(ROOT/'reference/expected_cross_period_decomposition.csv')
    assert len(d)==6
    assert d['closure'].abs().max() < 1e-10
