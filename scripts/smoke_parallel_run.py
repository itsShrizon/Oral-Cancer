"""
Driver for the parallelism smoke test. Runs three modes back to back:

  A: single process, NUM_WORKERS=0  (your current default)
  B: single process, NUM_WORKERS=4  (CPU-bottleneck check)
  C: two parallel processes, each NUM_WORKERS=4
     -> if total wall time of C is close to B, parallelism wins ~2x
     -> if total wall time of C >= 2 * B, GPU is saturated, no win

Each mode runs --iters training steps (default 40) on the real dataset.
"""

import argparse
import os
import subprocess
import sys
import time


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY   = os.path.join(ROOT, ".venv", "Scripts", "python.exe")
SMOKE = os.path.join(HERE, "smoke_parallel.py")


def run_blocking(tag, workers, iters, batch):
    cmd = [PY, SMOKE, "--tag", tag, "--workers", str(workers),
           "--iters", str(iters), "--batch", str(batch)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd)
    return time.perf_counter() - t0, r.returncode


def run_parallel(tags, workers, iters, batch):
    procs = []
    t0 = time.perf_counter()
    for tag in tags:
        cmd = [PY, SMOKE, "--tag", tag, "--workers", str(workers),
               "--iters", str(iters), "--batch", str(batch)]
        procs.append(subprocess.Popen(cmd))
    rcs = [p.wait() for p in procs]
    return time.perf_counter() - t0, rcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    print("=" * 70)
    print("Mode A: single process, NUM_WORKERS=0")
    print("=" * 70)
    a_t, _ = run_blocking("A", 0, args.iters, args.batch)
    print(f"\nMode A total wall time: {a_t:.2f}s\n")

    print("=" * 70)
    print("Mode B: single process, NUM_WORKERS=4")
    print("=" * 70)
    b_t, _ = run_blocking("B", 4, args.iters, args.batch)
    print(f"\nMode B total wall time: {b_t:.2f}s\n")

    print("=" * 70)
    print("Mode C: two parallel processes, NUM_WORKERS=4 each")
    print("=" * 70)
    c_t, _ = run_parallel(["C1", "C2"], 4, args.iters, args.batch)
    print(f"\nMode C total wall time (both finished): {c_t:.2f}s\n")

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  A (workers=0):           {a_t:.2f}s  for 1 process / {args.iters} iters")
    print(f"  B (workers=4):           {b_t:.2f}s  for 1 process / {args.iters} iters")
    print(f"  C (workers=4 x 2 par.):  {c_t:.2f}s  for 2 processes / {args.iters} iters each")

    # Equivalent throughput comparisons:
    seq_2x = 2 * b_t
    print(f"\n  Sequential 2x B = {seq_2x:.2f}s")
    print(f"  Parallel C      = {c_t:.2f}s")
    speedup = seq_2x / c_t if c_t > 0 else 0
    print(f"  Parallel speedup over sequential: {speedup:.2f}x")
    if speedup > 1.6:
        print("  -> Parallelism is a real win.")
    elif speedup > 1.2:
        print("  -> Modest parallel win; worth using if convenient.")
    else:
        print("  -> GPU is ~saturated. Run sequentially.")


if __name__ == "__main__":
    main()
