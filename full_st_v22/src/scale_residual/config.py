from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import tomllib

@dataclass(frozen=True)
class DataConfig:
    release: str
    raw_dir: Path
    processed_dir: Path
    base_url: str
    load_file: str
    wind_file: str
    bus_file: str
    branch_file: str
    gen_file: str

@dataclass(frozen=True)
class PaperConfig:
    main_year: int
    main_month: int
    history_start: str
    history_end: str
    use_paper_compat_generator_profile: bool
    co2_scope: str

@dataclass(frozen=True)
class SolverConfig:
    method: str
    feasibility_tol: float

@dataclass(frozen=True)
class OutputConfig:
    dir: Path
    write_hourly: bool
    write_plots: bool

@dataclass(frozen=True)
class VerificationConfig:
    mae_tolerance: float
    area_value_tolerance: float

@dataclass(frozen=True)
class Config:
    root: Path
    data: DataConfig
    paper: PaperConfig
    solver: SolverConfig
    output: OutputConfig
    verification: VerificationConfig


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    root = path.parent.parent
    with path.open("rb") as f:
        obj = tomllib.load(f)
    d = obj["data"]
    p = obj["paper"]
    s = obj["solver"]
    o = obj["output"]
    v = obj["verification"]
    return Config(
        root=root,
        data=DataConfig(
            release=d["release"],
            raw_dir=root / d["raw_dir"],
            processed_dir=root / d["processed_dir"],
            base_url=d["base_url"].rstrip("/"),
            load_file=d["load_file"],
            wind_file=d["wind_file"],
            bus_file=d["bus_file"],
            branch_file=d["branch_file"],
            gen_file=d["gen_file"],
        ),
        paper=PaperConfig(**p),
        solver=SolverConfig(**s),
        output=OutputConfig(
            dir=root / o["dir"],
            write_hourly=o["write_hourly"],
            write_plots=o["write_plots"],
        ),
        verification=VerificationConfig(**v),
    )
