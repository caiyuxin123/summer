# Final delivery baseline — 2026-09-09

Current baseline:

- paper: v8 实验资产恢复版；
- engineering package: v2.2；
- numerical fact source: `reference/expected_*.csv`；
- legacy evidence: `reference/legacy_*` only, not current paper facts.

Current headline results:

- Feb-2020 696 h: Direct-M 140.160 → Full-ST 1.117 kgCO2/MWh (−99.20%);
- Jan-target: 133.967 → 0.487 (−99.64%);
- Jun-target: 61.150 → 6.213 (−89.84%);
- Jan-wind Full-ST: 3.786 kgCO2/MWh;
- lossy AC identity closure: 1.66e-10; boundary exposure share 96.54%.

QA: bundled raw PASS; pytest 9 passed; current cross-period, wind and AC rerun PASS; automatic verify across all current result groups passed=true.
