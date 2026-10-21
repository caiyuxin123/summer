from pathlib import Path
import sys,subprocess,json
ROOT=Path(__file__).resolve().parents[1]
# PI candidates are deliberately isolated in separate processes to avoid threadpool interaction after sklearn fits.
C=[(0,16,0.001),(1,16,0.003),(2,32,0.001),(3,32,0.003)]
print(json.dumps({"candidates":C,"note":"use the frozen calibration/split/selection seed and assemble by MAE->RMSE->complexity->index"},indent=2))
