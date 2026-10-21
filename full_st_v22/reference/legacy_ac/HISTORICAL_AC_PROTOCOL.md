# Historical case39/case118 AC experiment protocol (provenance asset)

Recovered from an earlier manuscript. The original full experiment code and all raw intermediate files were not recovered into the v8 engineering baseline, so the following values are archived as **historical evidence only**.

## Historical setup

- MATPOWER New England case39: 39 buses / 10 generators / 46 branches.
- IEEE case118: 118 buses / 54 generators / 186 branches.
- Both were partitioned into five fixed topological regions; main partition seed 42.
- Main partition sizes: case39 = 8/13/5/9/4; case118 = 30/17/18/22/31.
- Normal and stress sequences: 720 h each for each system; 2,880 detailed AC snapshots in total.
- Pseudo-state reconstruction doubled AC power-flow solves to 5,760.
- Newton-Raphson tolerance 1e-8; 30 iterations, with cold-start fallback up to 60.
- Time-sequence random seed: 20260824.
- Historical lifecycle factors (gCO2e/kWh): wind 11, nuclear 12, hydro 24, PV 48, external 450, gas 490, coal 820.
- Active branch losses were explicit; midpoint transfer and half-loss allocation were used.
- Partition robustness seeds: 7, 21, 42, 84, 126; 72 h per partition/scenario.
- Participation perturbation: lognormal sigma = 0, 0.10, 0.20, 0.40.

## Historical numerical evidence

- max spatial-identity closure: 3.20e-10;
- max boundary/loss-decomposition closure: 2.47e-10;
- max rho(Q_C): 0.2006;
- min sigma_min(I-Q_C): 0.6660;
- max condition number: 2.164;
- boundary-exposure share: 96.4%--98.6%;
- direct boundary-flow correlation with scale error: 0.448--0.488;
- boundary-responsibility-exposure correlation: 0.778--0.996.

## Current v8 decision

The v8 paper does not claim the above historical numbers as newly rerun results. Instead, the current engineering package contains a self-contained, deterministic lossy-AC audit on the bundled RTS-GMLC 73-bus snapshot. Current paper facts come from `reference/expected_ac_audit.csv`; this directory preserves the earlier experiment as provenance only.
