from __future__ import annotations
from pathlib import Path
from urllib.request import Request, urlopen
import shutil
import pandas as pd

REQUIRED_KEYS = ("bus", "branch", "gen", "load", "wind")


def _url(base: str, rel: str) -> str:
    return f"{base}/{rel.lstrip('/')}"


def download_file(url: str, dst: Path, timeout: int = 60) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "scale-residual-repro/1.0"})
    with urlopen(req, timeout=timeout) as r, dst.open("wb") as f:
        shutil.copyfileobj(r, f)
    return dst


def expected_raw_paths(cfg) -> dict[str, Path]:
    return {
        "bus": cfg.data.raw_dir / "bus.csv",
        "branch": cfg.data.raw_dir / "branch.csv",
        "gen": cfg.data.raw_dir / "gen.csv",
        "load": cfg.data.raw_dir / "DAY_AHEAD_regional_Load.csv",
        "wind": cfg.data.raw_dir / "DAY_AHEAD_wind.csv",
    }


def ensure_rts_gmlc(cfg, force: bool = False) -> dict[str, Path]:
    paths = expected_raw_paths(cfg)
    rels = {
        "bus": cfg.data.bus_file,
        "branch": cfg.data.branch_file,
        "gen": cfg.data.gen_file,
        "load": cfg.data.load_file,
        "wind": cfg.data.wind_file,
    }
    for key, dst in paths.items():
        if force or not dst.exists():
            download_file(_url(cfg.data.base_url, rels[key]), dst)
    return paths


def read_regional_timeseries(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Year", "Month", "Day", "Period"}
    if not required.issubset(df.columns):
        raise ValueError(f"Unexpected RTS-GMLC time-series schema in {path}")
    dt = pd.to_datetime(dict(year=df.Year, month=df.Month, day=df.Day)) + pd.to_timedelta(df.Period - 1, unit="h")
    out = df.copy()
    out.index = pd.DatetimeIndex(dt, name="timestamp")
    return out


def slice_month(df: pd.DataFrame, year: int, month: int) -> pd.DataFrame:
    mask = (df.index.year == year) & (df.index.month == month)
    out = df.loc[mask].copy()
    if out.empty:
        raise ValueError(f"No rows for {year}-{month:02d}")
    return out


def slice_dates(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) + pd.Timedelta(hours=23)
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()
