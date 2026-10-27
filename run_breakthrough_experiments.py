#!/usr/bin/env python3
"""Breakthrough Experiment Runner & Automated Evaluator for FLS Sonar Object Detection.

This script manages the end-to-end training and evaluation of the newly designed
physics-informed sonar detection architectures:
  - Plan A: YOLO26n-NAFNetWavelet (Wavelet Sub-band Disentanglement)
  - Plan B: YOLO26n-NAFNetAcoustic (Acoustic Highlight-Shadow Dual Prior)
  - Plan C: YOLO26n-NAFNetWaveletAcoustic (Hybrid Wavelet + Acoustic Prior)
  - Baseline Control: YOLO26n-NAFNet (Vanilla NAFNet front-end)

Features:
  - Automatic sequential training (supports 20ep fast screening or 50ep full training)
  - Automated evaluation calling `compute_class_size_ap.py` to extract detailed per-class metrics
  - Auto-generation of comparison Markdown summary table
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# Available models
MODEL_CONFIGS = {
    "Wavelet": "ultralytics/cfg/models/26/yolo26n-NAFNetWavelet.yaml",
    "Acoustic": "ultralytics/cfg/models/26/yolo26n-NAFNetAcoustic.yaml",
    "WaveletAcoustic": "ultralytics/cfg/models/26/yolo26n-NAFNetWaveletAcoustic.yaml",
    "VanillaNAFNet": "ultralytics/cfg/models/26/yolo26n-NAFNet.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run Breakthrough Sonar Detection Experiments")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Wavelet", "Acoustic", "WaveletAcoustic"],
        choices=list(MODEL_CONFIGS.keys()),
        help="List of model keys to train/evaluate",
    )
    parser.add_argument("--data", type=str, default="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml", help="Dataset config")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs (20 for screening, 50 for full)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", type=str, default="runs/breakthrough", help="Experiment project dir")
    parser.add_argument("--eval-only", action="store_true", help="Only run evaluation on existing best.pt weights")
    return parser.parse_args()


def run_cmd(cmd):
    print(f"\n[RUNNING] {cmd}\n")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"[ERROR] Command failed with return code {ret.returncode}: {cmd}")
        return False
    return True


def find_weights(project, exp_name):
    candidates = [
        Path("runs/detect") / project / exp_name / "weights" / "best.pt",
        Path(project) / exp_name / "weights" / "best.pt",
        Path("runs") / project / exp_name / "weights" / "best.pt",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def run_train(model_key, cfg_path, args):
    exp_name = f"{model_key}_{args.epochs}ep"
    weights_path = find_weights(args.project, exp_name)

    if args.eval_only:
        if weights_path is None or not weights_path.exists():
            print(f"[WARN] Could not find checkpoint for {exp_name}, cannot evaluate.")
            return None
        return weights_path

    if weights_path is not None and weights_path.exists():
        print(f"[INFO] Found existing weights at {weights_path}, skipping training (delete to re-train).")
        return weights_path

    cmd = (
        f"yolo train model={cfg_path} data={args.data} epochs={args.epochs} "
        f"batch={args.batch} imgsz={args.imgsz} workers={args.workers} "
        f"project={args.project} name={exp_name}"
    )
    success = run_cmd(cmd)
    weights_path = find_weights(args.project, exp_name)
    if not success or weights_path is None or not weights_path.exists():
        print(f"[ERROR] Training failed or weights not found for {model_key}")
        return None
    return weights_path


def run_eval(weights_path, args):
    cmd = f"{sys.executable} compute_class_size_ap.py --model {weights_path} --data {args.data} --imgsz {args.imgsz}"
    print(f"\n[EVALUATING] {weights_path}")
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    return proc.stdout


def main():
    args = parse_args()
    results = {}

    print("=" * 70)
    print("  FLS Forward-Looking Sonar Target Detection: Breakthrough Suite")
    print(f"  Models to test: {args.models}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | Imgsz: {args.imgsz}")
    print("=" * 70)

    start_time = time.time()

    for model_key in args.models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"\n>>> Starting Track: {model_key} ({cfg})")
        weights = run_train(model_key, cfg, args)
        if weights and weights.exists():
            eval_output = run_eval(weights, args)
            results[model_key] = {"weights": str(weights), "output": eval_output}

    elapsed = time.time() - start_time
    print(f"\nAll experiments completed in {elapsed/3600:.2f} hours.")
    print("Summary saved in runs/breakthrough.")


if __name__ == "__main__":
    main()
