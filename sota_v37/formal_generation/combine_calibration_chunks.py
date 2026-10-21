from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--chunks-dir',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
    files=sorted(Path(a.chunks_dir).glob('calibration_chunk_*.npz'))
    if not files: raise FileNotFoundError('no chunks')
    records=[]; graph=None
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            keys=set(z.files)
            exp={'scenario_index','x_month','x_history','graph','y'}
            if keys!=exp: raise ValueError(f'{p}: {keys}')
            g=np.asarray(z['graph'],dtype=np.float32)
            if graph is None: graph=g
            elif not np.array_equal(graph,g): raise ValueError(f'graph mismatch {p}')
            for j,i in enumerate(z['scenario_index'].astype(int).tolist()):
                records.append((i,z['x_month'][j].copy(),z['x_history'][j].copy(),z['y'][j].copy()))
    records.sort(key=lambda q:q[0])
    idx=[q[0] for q in records]
    if idx!=list(range(120)): raise ValueError(f'expected exactly indices 0..119, got {idx}')
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,
        x_month=np.stack([q[1] for q in records]).astype(np.float32),
        x_history=np.stack([q[2] for q in records]).astype(np.float32),
        graph=graph,
        y=np.stack([q[3] for q in records]).astype(np.float32))
    print(out)

if __name__=='__main__': main()
