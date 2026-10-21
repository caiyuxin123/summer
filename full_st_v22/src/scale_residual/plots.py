from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_natural_month(outdir: Path) -> None:
    dec=pd.read_csv(outdir/"residual_decomposition.csv")
    summ=pd.read_csv(outdir/"natural_month_summary.csv")
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    x=dec.area.astype(str)
    ax.bar(x,dec.Delta_spatial,label="Spatial residual")
    ax.bar(x,dec.Delta_temporal,bottom=dec.Delta_spatial,label="Temporal residual")
    ax.axhline(0,linewidth=0.8)
    ax.set_xlabel("Area"); ax.set_ylabel("kgCO$_2$/MWh"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(outdir/"residual_decomposition.png",dpi=220); plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    ax.bar(summ.method,summ.MAE)
    ax.set_ylabel("MAE (kgCO$_2$/MWh)"); ax.tick_params(axis="x",rotation=20)
    fig.tight_layout(); fig.savefig(outdir/"method_mae.png",dpi=220); plt.close(fig)


def plot_wind(outdir: Path) -> None:
    summ=pd.read_csv(outdir/"wind_summary.csv")
    fig,ax=plt.subplots(figsize=(6.6,3.6))
    ax.bar(summ.method,summ.MAE)
    ax.set_ylabel("MAE (kgCO$_2$/MWh)"); ax.tick_params(axis="x",rotation=20)
    fig.tight_layout(); fig.savefig(outdir/"wind_ablation_mae.png",dpi=220); plt.close(fig)
