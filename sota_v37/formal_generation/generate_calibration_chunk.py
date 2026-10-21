from __future__ import annotations
"""Resumable sequential chunk generator for environments with bounded command runtime."""
import argparse, json
from pathlib import Path
import numpy as np
from .apply_scenario import perturb_base_load
from .generate_calibration_from_backend import load_backend, REQUIRED_KEYS


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base-history-npy',required=True)
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--backend',required=True)
    ap.add_argument('--start',type=int,required=True)
    ap.add_argument('--stop',type=int,required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    base=np.load(a.base_history_npy)
    rows=json.loads(Path(a.manifest).read_text(encoding='utf-8'))
    if len(rows)!=120 or base.shape!=(168,3): raise ValueError('frozen design mismatch')
    if not (0<=a.start<a.stop<=120): raise ValueError('invalid range')
    backend=load_backend(a.backend)
    samples=[]
    for i in range(a.start,a.stop):
        r=rows[i]
        target=perturb_base_load(base,r,'target')
        context=perturb_base_load(base,r,'context')
        s=backend.build_calibration_sample(target,context)
        if set(s)!=REQUIRED_KEYS: raise ValueError(f'bad keys at {i}: {set(s)}')
        samples.append(s)
    graph=np.asarray(samples[0]['graph'],dtype=np.float32)
    for s in samples:
        if not np.array_equal(np.asarray(s['graph'],dtype=np.float32),graph): raise ValueError('graph mismatch')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,
        scenario_index=np.arange(a.start,a.stop,dtype=np.int32),
        x_month=np.stack([s['x_month'] for s in samples]).astype(np.float32),
        x_history=np.stack([s['x_history'] for s in samples]).astype(np.float32),
        graph=graph,
        y=np.stack([s['y'] for s in samples]).astype(np.float32))
    print(f'wrote {out} scenarios [{a.start},{a.stop})')

if __name__=='__main__': main()
