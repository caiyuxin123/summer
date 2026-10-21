from __future__ import annotations
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

FIELDS = ['scenario_id','target_shift_h','context_shift_h'] + \
         [f'target_area_mult_{a}' for a in range(1,4)] + [f'context_area_mult_{a}' for a in range(1,4)]
for role in ('target','context'):
    for a in range(1,4):
        for k in range(1,3):
            FIELDS += [f'{role}_a{a}_sin{k}_amp',f'{role}_a{a}_sin{k}_period_h',f'{role}_a{a}_sin{k}_phase_rad']


def _one_spec(rng, prefix):
    d={f'{prefix}_shift_h':int(rng.integers(-24,25))}
    for a in range(1,4):
        d[f'{prefix}_area_mult_{a}']=float(rng.uniform(0.88,1.12))
        for k in range(1,3):
            d[f'{prefix}_a{a}_sin{k}_amp']=float(rng.uniform(-0.08,0.08))
            d[f'{prefix}_a{a}_sin{k}_period_h']=float(rng.uniform(48.0,168.0))
            d[f'{prefix}_a{a}_sin{k}_phase_rad']=float(rng.uniform(0.0,2*np.pi))
    return d


def make_manifest(n=120, seed=20260929):
    rng=np.random.default_rng(int(seed)); rows=[]
    for i in range(int(n)):
        r={'scenario_id':f'cal_{i:03d}'}
        r.update(_one_spec(rng,'target'))
        # Independent draw: target hourly shape is never copied into the context input.
        r.update(_one_spec(rng,'context'))
        rows.append(r)
    return rows


def validate(rows):
    assert len(rows)==120
    ids=[r['scenario_id'] for r in rows]; assert len(set(ids))==120
    for r in rows:
        for prefix in ('target','context'):
            assert -24 <= int(r[f'{prefix}_shift_h']) <= 24
            for a in range(1,4):
                assert 0.88 <= float(r[f'{prefix}_area_mult_{a}']) <= 1.12
                for k in range(1,3):
                    assert abs(float(r[f'{prefix}_a{a}_sin{k}_amp'])) <= 0.08 + 1e-12
                    assert 48 <= float(r[f'{prefix}_a{a}_sin{k}_period_h']) <= 168
    return True


def write(rows, outdir):
    outdir=Path(outdir); outdir.mkdir(parents=True,exist_ok=True)
    js=outdir/'calibration_120_scenario_manifest.json'
    cs=outdir/'calibration_120_scenario_manifest.csv'
    js.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with cs.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    h=hashlib.sha256(js.read_bytes()).hexdigest()
    (outdir/'calibration_120_manifest_sha256.txt').write_text(h+'  '+js.name+'\n',encoding='utf-8')
    return h


def main():
    p=argparse.ArgumentParser(); p.add_argument('--outdir',required=True); p.add_argument('--seed',type=int,default=20260929); a=p.parse_args()
    rows=make_manifest(120,a.seed); validate(rows); print(write(rows,a.outdir))
if __name__=='__main__': main()
