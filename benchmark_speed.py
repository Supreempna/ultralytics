#!/usr/bin/env python3
"""Measure inference speed (forward latency + FPS) for the NAFNet ablation models.

Measures the raw model forward on a single image (batch=1, imgsz=640) after warmup.
This isolates the GPU compute of NAFNet + backbone + head, excluding CPU pre/post-processing
(letterbox, NMS). Reported as latency (ms/image) and FPS.

Usage:
    python benchmark_speed.py \
        --models runs/detect/<baseline>/weights/best.pt \
                 runs/detect/<nafnet>/weights/best.pt \
                 runs/ablation/m16b1/weights/best.pt \
                 runs/ablation/m8b2/weights/best.pt \
                 runs/ablation/m8b1/weights/best.pt \
                 runs/detect/<nafnetfull>/weights/best.pt \
        --device 0
"""

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def measure(weights, imgsz, device, warmup=30, iters=200):
    """Return (latency_ms, fps) for a single-image forward pass."""
    net = YOLO(weights).model.to(device)
    net.eval()
    x = torch.zeros(1, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(warmup):  # warmup (kernel compile, allocator)
            net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters

    return dt * 1000.0, 1.0 / dt  # latency (ms), FPS


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="paths to best.pt files")
    ap.add_argument("--imgsz", type=int, default=640, help="input size (default 640)")
    ap.add_argument("--device", default="0", help="device index (default 0) or 'cpu'")
    ap.add_argument("--warmup", type=int, default=30, help="warmup iterations")
    ap.add_argument("--iters", type=int, default=200, help="timed iterations")
    args = ap.parse_args()

    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda", int(args.device))

    print(f"device={device}, imgsz={args.imgsz}, batch=1, iters={args.iters}")
    print("")
    print(f"{'model':<28} {'latency(ms)':>12} {'FPS':>10}")
    print("-" * 52)

    for p in args.models:
        name = Path(p).resolve().parent.parent.name  # runs/<project>/<name>/weights/best.pt -> <name>
        latency, fps = measure(p, args.imgsz, device, args.warmup, args.iters)
        print(f"{name:<28} {latency:>12.2f} {fps:>10.1f}")


if __name__ == "__main__":
    main()
