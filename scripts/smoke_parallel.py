"""
Parallelism smoke test — does running 2 training processes in parallel on
one RTX 4060 Ti actually save wall-clock time, or just split a saturated
GPU into halves?

Runs N training iterations (forward + backward + optimizer step) of Custom
EfficientNet V2 end-to-end (real dataloader, real images, real GPU) and
reports throughput. Used three ways:

  Mode A: single process, NUM_WORKERS=0   (current default)
  Mode B: single process, NUM_WORKERS=4   (CPU-bottleneck check)
  Mode C: launched twice in parallel by smoke_parallel_run.py

If parallelism helps, mode C's per-process time should be close to mode B's
single-process time. If the GPU is saturated, mode C will take ~2x longer
per process than mode B and total wall time is unchanged.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs import config
from utils.common import set_seed, get_device
from data.transforms import train_transform
from data.dataset import (OralPathologyDataset, load_dataset1_split,
                          load_dataset2_split)
from models.architecture import MultiTaskOralClassifier
from models.loss import MultiTaskLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters",   type=int, default=40,
                    help="Number of training iterations to time.")
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader num_workers.")
    ap.add_argument("--batch",   type=int, default=32)
    ap.add_argument("--tag",     type=str, default="A",
                    help="Label printed in output (A/B/C1/C2).")
    args = ap.parse_args()

    set_seed()
    device = get_device()
    if device.type != "cuda":
        print("CUDA not available — this benchmark is GPU-only."); sys.exit(1)

    d1p, d1b, d1s = load_dataset1_split('train')
    d2p, d2b, d2s = load_dataset2_split('train')
    ds = OralPathologyDataset(d1p + d2p, d1b + d2b, d1s + d2s,
                              transform=train_transform)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=(args.workers > 0))

    model = MultiTaskOralClassifier(backbone="custom_efficientnet_v2").to(device)
    model.train()
    criterion = MultiTaskLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup — first batch always pays JIT / cuDNN benchmark cost.
    it = iter(loader)
    imgs, yb, ys = next(it)
    imgs, yb, ys = imgs.to(device), yb.to(device), ys.to(device)
    for _ in range(3):
        optimizer.zero_grad()
        ob, os_ = model(imgs)
        loss, _, _ = criterion(ob, os_, yb, ys)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()

    # Timed loop — recreate iterator to include dataloader cost in timing.
    t0 = time.perf_counter()
    n = 0
    it = iter(loader)
    while n < args.iters:
        try:
            imgs, yb, ys = next(it)
        except StopIteration:
            it = iter(loader)
            imgs, yb, ys = next(it)
        imgs = imgs.to(device, non_blocking=True)
        yb   = yb.to(device, non_blocking=True)
        ys   = ys.to(device, non_blocking=True)
        optimizer.zero_grad()
        ob, os_ = model(imgs)
        loss, _, _ = criterion(ob, os_, yb, ys)
        loss.backward()
        optimizer.step()
        n += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    samples = n * args.batch
    print(f"[{args.tag}] iters={n} batch={args.batch} workers={args.workers} "
          f"elapsed={elapsed:.2f}s | {n/elapsed:.2f} it/s | "
          f"{samples/elapsed:.1f} samples/s")


if __name__ == "__main__":
    main()
