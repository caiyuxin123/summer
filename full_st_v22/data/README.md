# Data directory — v8 experiment-asset recovery baseline

## Bundled raw-source data

`data/raw/` contains the RTS-GMLC release-0.2 official-source rows required by the current reproducible experiments. The default reproduction path is offline.

- `bus.csv` — full SourceData file, 73 rows;
- `branch.csv` — full SourceData file, 120 rows;
- `gen.csv` — full SourceData file, 158 rows;
- `DAY_AHEAD_regional_Load.csv` — Jan 1–14 + Feb 1–29 + Jun 1–14, **1368 hourly rows**;
- `DAY_AHEAD_wind.csv` — Jan 1–14, 336 hourly rows.

Time-series values are official RTS-GMLC source rows. The only local transformation is selecting the experiment-required dates and concatenating them under the original header.

## Experiment windows

- Natural month: history Jan 1–7; target Feb 1–29 (696 h).
- Current cross-period validation: one frozen history Jan 1–7; targets Jan 8–14 and Jun 8–14 (168 h each).
- Jan-wind: history Jan 1–7; target Jan 8–14, official wind availability.
- Lossy-AC audit: static RTS-GMLC bus/branch snapshot; no additional time series required.

## Integrity

`raw_data_manifest.csv` records source path, release, included scope, row count, byte size, SHA256 and upstream blob SHA1. `MANIFEST.sha256` directly hashes bundled raw files.

Run `python scripts/validate_bundled_raw.py` before reproduction.
