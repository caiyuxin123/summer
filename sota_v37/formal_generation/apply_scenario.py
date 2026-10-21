from __future__ import annotations
import numpy as np

def perturb_base_load(base_168x3: np.ndarray, spec: dict, role: str) -> np.ndarray:
    """Apply only the frozen, target-external calibration perturbation. No target-period data are accepted."""
    x=np.asarray(base_168x3,dtype=float)
    if x.shape != (168,3): raise ValueError(f'base load must be (168,3), got {x.shape}')
    shift=int(spec[f'{role}_shift_h'])
    x=np.roll(x,shift=shift,axis=0).copy()
    t=np.arange(168,dtype=float)
    for j,a in enumerate(range(1,4)):
        shape=np.ones(168,dtype=float)
        for k in range(1,3):
            amp=float(spec[f'{role}_a{a}_sin{k}_amp'])
            per=float(spec[f'{role}_a{a}_sin{k}_period_h'])
            ph=float(spec[f'{role}_a{a}_sin{k}_phase_rad'])
            shape += amp*np.sin(2*np.pi*t/per + ph)
        x[:,j] *= float(spec[f'{role}_area_mult_{a}']) * shape
    if not np.isfinite(x).all() or np.any(x<=0): raise ValueError('invalid perturbed load')
    return x
