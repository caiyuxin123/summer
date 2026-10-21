from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def _count(name):
    with (RAW/name).open(newline="", encoding="utf-8-sig") as f:
        return sum(1 for _ in csv.DictReader(f))


def test_bundled_raw_coverage():
    assert _count("bus.csv") == 73
    assert _count("branch.csv") == 120
    assert _count("gen.csv") == 158
    assert _count("DAY_AHEAD_regional_Load.csv") == 1368
    assert _count("DAY_AHEAD_wind.csv") == 336
