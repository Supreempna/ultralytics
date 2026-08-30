"""Remap a stage-1 (SpeckleNoise + NAFNet) checkpoint onto a stage-2 (NAFNet-only) architecture.

Background
----------
Stage 1 trains ``yolo26n-NAFNet-SpeckleNoise.yaml`` (SpeckleNoise at layer 0, NAFNet at layer 1).
Stage 2 fine-tunes ``yolo26n-NAFNet.yaml`` (NAFNet at layer 0, no SpeckleNoise).

Because removing the layer-0 SpeckleNoise shifts every subsequent layer index by -1,
a naive ``model.load(stage1_best.pt)`` matches ZERO keys (ultralytics' ``intersect_dicts``
is key-name based) and silently leaves the stage-2 model randomly initialized.

This script shifts the ``model.<N>.*`` keys by -1 so stage-1 weights transfer 1:1 into the
stage-2 architecture. SpeckleNoise itself has no parameters, so there is nothing to drop.

Usage
-----
    python remap_stage1_to_stage2.py --stage1 runs/detect/train/weights/best.pt \
        --stage2 ultralytics/cfg/models/26/yolo26n-NAFNet.yaml \
        --out runs/detect/train/weights/stage2_init.pt

Then fine-tune stage 2:
    yolo train model=runs/detect/train/weights/stage2_init.pt data=your_data.yaml epochs=...
"""

import argparse
import sys

import torch

from ultralytics.nn.tasks import DetectionModel


def remap_state_dict(state_dict):
    """Shift ``model.<N>.*`` keys by -1 to drop the removed layer-0 module."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            _, idx, rest = key.split(".", 2)
            idx = int(idx)
            if idx <= 0:
                raise ValueError(f"Unexpected layer-0 parameter key in stage-1 checkpoint: {key}")
            remapped[f"model.{idx - 1}.{rest}"] = value
        else:
            remapped[key] = value
    return remapped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1", required=True, help="path to stage-1 best.pt (with SpeckleNoise)")
    parser.add_argument("--stage2", required=True, help="path to stage-2 yaml (NAFNet-only)")
    parser.add_argument("--out", required=True, help="output path for the stage-2 initial checkpoint")
    args = parser.parse_args()

    ckpt = torch.load(args.stage1, map_location="cpu", weights_only=False)
    stage1 = ckpt.get("ema") or ckpt["model"]  # prefer EMA weights when present

    # Confirm the stage-1 architecture actually starts with SpeckleNoise (sanity check)
    first_module = stage1.yaml["backbone"][0][2]
    if first_module != "SpeckleNoise":
        print(f"[warning] stage-1 first backbone module is '{first_module}', expected 'SpeckleNoise'", file=sys.stderr)

    nc = stage1.yaml.get("nc", 80)
    ch = stage1.yaml.get("channels", 3)

    # Build stage-2 architecture with matching nc/ch so the Detect head aligns
    stage2 = DetectionModel(args.stage2, ch=ch, nc=nc, verbose=False)

    remapped = remap_state_dict(stage1.float().state_dict())
    result = stage2.load_state_dict(remapped, strict=False)

    total = len(stage2.state_dict())
    matched = total - len(result.missing_keys)
    print(f"keys transferred: {matched}/{total}")
    if result.missing_keys:
        print("missing keys:", result.missing_keys)
    if result.unexpected_keys:
        print("unexpected keys:", result.unexpected_keys)

    # Minimal checkpoint loadable by YOLO() and by `yolo train model=...`
    torch.save({"model": stage2, "epoch": -1}, args.out)
    print("saved:", args.out)


if __name__ == "__main__":
    main()
