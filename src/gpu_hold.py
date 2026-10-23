"""Temporarily reserve GPU memory while code is being written; exits when RELEASE file appears or after a timeout."""
import os, sys, time, torch
gb, release, tmax = float(sys.argv[1]), sys.argv[2], float(sys.argv[3])
x = torch.empty(int(gb * 2**30), dtype=torch.uint8, device="cuda")
t0 = time.time()
while not os.path.exists(release) and time.time() - t0 < tmax:
    time.sleep(1)
