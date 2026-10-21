from __future__ import annotations
import numpy as np

def error_metrics(error: np.ndarray) -> dict[str,float]:
    e=np.asarray(error,float)
    return {
        "MAE": float(np.mean(np.abs(e))),
        "RMSE": float(np.sqrt(np.mean(e**2))),
        "MaxAE": float(np.max(np.abs(e))),
    }
