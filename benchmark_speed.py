#!/usr/bin/env python3
"""Measure inference speed (forward latency + FPS) for YOLO models.

Measures the raw model forward pass on a single image (batch=1, imgsz=640) after warmup.
This isolates the GPU compute of NAFNet + backbone + neck + head, excluding CPU
pre/post-processing (letterbox, NMS). Supports both trained checkpoints (.pt)
and architecture configs (.yaml), and both FP32 and FP16 (half precision).

Usage Examples:
    # 1. Benchmark trained weights on GPU 0 in FP16 (recommended for edge deployment):
    python benchmark_speed.py --models runs/detect/breakthrough/*/weights/best.pt --device 0 --half

    # 2. Benchmark specific model YAMLs or weights in FP32:
    python benchmark_speed.py --models ultralytics/cfg/models/26/yolo26.yaml \
                                      ultralytics/cfg/models/26/yolo26n-NAFNet.yaml \
                                      ultralytics/cfg/models/26/yolo26n-NAFNetWavelet.yaml \
                                      ultralytics/cfg/models/26/yolo26n-NAFNetWavelet-DySample.yaml \
                              --device 0

    # 3. Benchmark on CPU:
    python benchmark_speed.py --models ultralytics/cfg/models/26/yolo26n-NAFNetWavelet-DySample.yaml --device cpu
"""

import argparse
import time
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


def resolve_model_name(path_str):
    """Return a clean, human-readable name for weights or yaml files."""
    p = Path(path_str)
    if p.suffix == ".pt":
        if p.parent.name == "weights":
            return p.parent.parent.name
        return p.stem
    return p.stem


def measure_forward(model_path, imgsz, device, is_half=False, warmup=50, iters=300):
    """Return (latency_mean_ms, latency_std_ms, fps, params_m, flops_g)."""
    # Load model
    m = YOLO(model_path)
    net = m.model.to(device)
    net.eval()
    if is_half and device.type == "cuda":
        net.half()

    dtype = torch.float16 if (is_half and device.type == "cuda") else torch.float32
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    # Compute parameters
    params = sum(p.numel() for p in net.parameters())
    params_m = params / 1e6

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = net(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Timed iterations
        latencies = []
        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            for _ in range(iters):
                starter.record()
                _ = net(x)
                ender.record()
                torch.cuda.synchronize()
                latencies.append(starter.elapsed_time(ender))  # milliseconds
        else:
            for _ in range(iters):
                t0 = time.perf_counter()
                _ = net(x)
                dt = (time.perf_counter() - t0) * 1000.0  # ms
                latencies.append(dt)

    latencies = np.array(latencies)
    # Exclude outliers (e.g. top and bottom 2%)
    p_low, p_high = np.percentile(latencies, [2, 98])
    filtered = latencies[(latencies >= p_low) & (latencies <= p_high)]

    mean_ms = float(np.mean(filtered))
    std_ms = float(np.std(filtered))
    fps = 1000.0 / mean_ms

    return mean_ms, std_ms, fps, params_m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", required=True, help="paths to .pt weights or .yaml files")
    ap.add_argument("--imgsz", type=int, default=640, help="input image size (default 640)")
    ap.add_argument("--device", default="0", help="device index (e.g. 0) or 'cpu'")
    ap.add_argument("--half", action="store_true", help="use FP16 half precision (recommended for CUDA)")
    ap.add_argument("--warmup", type=int, default=50, help="warmup iterations (default 50)")
    ap.add_argument("--iters", type=int, default=300, help="timed iterations (default 300)")
    args = ap.parse_args()

    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda", int(args.device))
    dev_name = torch.cuda.get_device_name(device.index if device.index is not None else 0) if device.type == "cuda" else "CPU"
    prec_str = "FP16" if (args.half and device.type == "cuda") else "FP32"

    print("=" * 75)
    print(" Benchmarking Latency & FPS for Edge Deployment")
    print(f" Target Device   : {device} ({dev_name})")
    print(f" Precision       : {prec_str}")
    print(f" Resolution      : {args.imgsz}x{args.imgsz} (batch=1)")
    print(f" Benchmark Specs : Warmup={args.warmup}, Iterations={args.iters}")
    print("=" * 75)
    print(f"{'Model Name':<32} {'Params':>10} {'Latency (ms)':>18} {'FPS':>10}")
    print("-" * 75)

    results = []
    for p in args.models:
        name = resolve_model_name(p)
        try:
            mean_ms, std_ms, fps, params_m = measure_forward(
                p, args.imgsz, device, is_half=args.half, warmup=args.warmup, iters=args.iters
            )
            lat_str = f"{mean_ms:6.2f} ± {std_ms:4.2f}"
            print(f"{name:<32} {params_m:8.3f}M {lat_str:>18} {fps:9.1f}")
            results.append((name, params_m, mean_ms, std_ms, fps))
        except Exception as e:
            print(f"{name:<32} [ERROR]: {e}")

    print("=" * 75)
    print("\n### Markdown Table for Paper / Knowledge Base:\n")
    print(f"| Model | Precision | Params (M) | Latency (ms) | Throughput (FPS) |")
    print(f"| :--- | :---: | :---: | :---: | :---: |")
    for name, params_m, mean_ms, std_ms, fps in results:
        print(f"| **{name}** | {prec_str} | {params_m:.3f}M | {mean_ms:.2f} ± {std_ms:.2f} ms | **{fps:.1f}** |")


if __name__ == "__main__":
    main()
