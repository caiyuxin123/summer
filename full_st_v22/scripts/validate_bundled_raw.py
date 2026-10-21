from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def read_rows(name: str):
    with (RAW / name).open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def validate_time_series(name: str, expected_dates: dict[tuple[int, int], set[int]], expected_rows: int) -> None:
    rows = read_rows(name)
    assert len(rows) == expected_rows, (name, len(rows), expected_rows)
    keys = [(int(r["Year"]), int(r["Month"]), int(r["Day"]), int(r["Period"])) for r in rows]
    assert len(keys) == len(set(keys)), f"duplicate timestamps in {name}"
    periods = defaultdict(set)
    for y, m, d, p in keys:
        periods[(y, m, d)].add(p)
    for (y, m), days in expected_dates.items():
        for d in days:
            assert periods[(y, m, d)] == set(range(1, 25)), f"incomplete day {y}-{m:02d}-{d:02d} in {name}"


def main() -> None:
    for name in ["bus.csv", "branch.csv", "gen.csv", "DAY_AHEAD_regional_Load.csv", "DAY_AHEAD_wind.csv"]:
        assert (RAW / name).is_file(), f"missing bundled raw file: {name}"
    assert len(read_rows("bus.csv")) == 73
    assert len(read_rows("branch.csv")) == 120
    assert len(read_rows("gen.csv")) == 158
    validate_time_series(
        "DAY_AHEAD_regional_Load.csv",
        {(2020, 1): set(range(1, 15)), (2020, 2): set(range(1, 30)), (2020, 6): set(range(1, 15))},
        1368,
    )
    validate_time_series(
        "DAY_AHEAD_wind.csv",
        {(2020, 1): set(range(1, 15))},
        336,
    )
    print("Bundled raw-data validation: PASS")
    print("  bus=73, branch=120, gen=158")
    print("  load=1368 hourly rows (Jan 1-14 + Feb 1-29 + Jun 1-14, 2020)")
    print("  wind=336 hourly rows (Jan 1-14, 2020)")
    print("  timestamps unique; each included day has Period 1..24")


if __name__ == "__main__":
    main()
