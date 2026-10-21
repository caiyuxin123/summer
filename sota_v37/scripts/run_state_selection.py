from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import argparse,json,torch
torch.set_num_threads(1); torch.set_num_interop_threads(1)
from state_comparators.selection import run
ap=argparse.ArgumentParser();ap.add_argument('--calibration',required=True);ap.add_argument('--states',required=True);ap.add_argument('--grid',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--projection',required=True);ap.add_argument('--out',required=True);a=ap.parse_args(); print(json.dumps(run(a.calibration,a.states,a.grid,a.out,a.protocol,a.projection),indent=2))
