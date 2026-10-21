from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd

def mkdir(path: Path) -> Path:
    path.mkdir(parents=True,exist_ok=True); return path

def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    def conv(x):
        if isinstance(x,(np.floating,np.integer)): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x))
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=conv),encoding="utf-8")
