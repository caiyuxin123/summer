#!/usr/bin/env python3
import subprocess,sys
subprocess.check_call([sys.executable,"-m","pytest","-q"])
print("Smoke test passed: network, CEF, and hidden-state LP are operational.")
