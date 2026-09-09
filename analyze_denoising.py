#!/usr/bin/env python3
"""Analyze NAFNet denoising: ENL (speckle-noise metric) + side-by-side visualization.

For each image this script:
  1. preprocesses (resize to imgsz, normalize to [0,1]);
  2. runs the NAFNet front-end (model.model[0]) to get the denoised output;
  3. computes ENL = (mean/std)^2 on the input and the output (whole image + patch median);
  4. saves an input | output | residual(x5) comparison image.

ENL is the standard speckle-noise metric for SAR/sonar: higher ENL = less noise. If the
output ENL is consistently higher than the input ENL, NAFNet is genuinely suppressing noise.

Usage:
    python analyze_denoising.py --model runs/detect/<nafnet>/weights/best.pt \
        --source ultralytics/dataset/FLS_Detection_YOLO/images/val \
        --out runs/denoise_viz --device 0 --num 20
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(src):
    """Return a list of image paths from a directory or a list of files."""
    src = Path(src)
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [src]


def enl(x):
    """ENL = (mean / std)^2 for a 2D float array."""
    x = np.asarray(x, dtype=np.float32)
    return float((x.mean() / (x.std() + 1e-6)) ** 2)


def patch_median_enl(x, patch=64):
    """Robust ENL: median over non-overlapping patches (ignores target outliers)."""
    h, w = x.shape[:2]
    vals = []
    for i in range(0, h - patch + 1, patch):
        for j in range(0, w - patch + 1, patch):
            p = x[i : i + patch, j : j + patch].astype(np.float32)
            if p.std() > 1e-6:
                vals.append((p.mean() / p.std()) ** 2)
    return float(np.median(vals)) if vals else 0.0


def to_gray(img):
    """Convert a (H,W,3) or (H,W) array to a float32 grayscale array."""
    img = np.asarray(img)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to NAFNet model best.pt")
    ap.add_argument("--source", required=True, help="image directory or a single image path")
    ap.add_argument("--out", default="runs/denoise_viz", help="output dir for comparison images")
    ap.add_argument("--imgsz", type=int, default=640, help="resize size")
    ap.add_argument("--device", default="0", help="device index or 'cpu'")
    ap.add_argument("--num", type=int, default=20, help="max number of images to process")
    args = ap.parse_args()

    device = torch.device("cpu") if args.device == "cpu" else torch.device("cuda", int(args.device))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model).to(device)
    nafnet = model.model[0]  # NAFNet is layer 0 in yolo26n-NAFNet.yaml
    nafnet.eval()

    images = list_images(args.source)[: args.num]
    print(f"processing {len(images)} images, NAFNet = {type(nafnet).__name__}\n")
    print(f"{'image':<24} {'ENL_in':>8} {'ENL_out':>9} {'med_in':>8} {'med_out':>9}")

    enl_ins, enl_outs, med_ins, med_outs = [], [], [], []
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[skip] cannot read {img_path}")
            continue

        x = cv2.resize(img, (args.imgsz, args.imgsz)).astype(np.float32) / 255.0
        x_t = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            y_t = nafnet(x_t)
        y = y_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        y = np.clip(y, 0.0, 1.0)

        g_in, g_out = to_gray(x), to_gray(y)
        e_in, e_out = enl(g_in), enl(g_out)
        m_in, m_out = patch_median_enl(g_in), patch_median_enl(g_out)
        enl_ins.append(e_in); enl_outs.append(e_out); med_ins.append(m_in); med_outs.append(m_out)

        print(f"{img_path.stem:<24} {e_in:>8.2f} {e_out:>9.2f} {m_in:>8.2f} {m_out:>9.2f}")

        # side-by-side: input | output | residual (x5)
        inp_u8 = (x * 255).astype(np.uint8)
        out_u8 = (y * 255).astype(np.uint8)
        diff_u8 = (np.clip(np.abs(x - y) * 5, 0, 1) * 255).astype(np.uint8)
        side = np.hstack([inp_u8, out_u8, diff_u8])
        cv2.imwrite(str(out_dir / f"{img_path.stem}_cmp.png"), side)

    if enl_ins:
        print("\n" + "-" * 52)
        print(f"mean  ENL : {np.mean(enl_ins):.2f}  ->  {np.mean(enl_outs):.2f}   ({100*(np.mean(enl_outs)/np.mean(enl_ins)-1):+.1f}%)")
        print(f"median ENL : {np.mean(med_ins):.2f}  ->  {np.mean(med_outs):.2f}")
        print(f"comparison images saved to: {out_dir}")
        if np.mean(enl_outs) > np.mean(enl_ins):
            print("=> ENL 提升，噪声被抑制（去噪有效）。")
        else:
            print("=> ENL 未提升，去噪效果存疑。")


if __name__ == "__main__":
    main()
