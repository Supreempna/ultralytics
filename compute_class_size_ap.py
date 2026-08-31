#!/usr/bin/env python3
"""Per-class x per-size COCO AP table for a YOLO-format dataset.

Extends ``compute_size_ap.py``: besides the overall size-stratified AP, this script
runs COCO evaluation once per category (``params.catIds=[cat]``) and reports, for each
class, AP_small / AP_medium / AP_large together with the ground-truth instance counts in
each size bin — so you can see WHICH class x size combination is the real bottleneck.

COCO size thresholds (ground-truth bbox area in original-image pixels):
    small  : < 32^2 = 1024 px^2
    medium : 32^2 <= area < 96^2 = 9216 px^2
    large  : >= 96^2

Usage
-----
    python compute_class_size_ap.py --model runs/detect/train/weights/best.pt \
        --data ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml --imgsz 640

Note: bins with very few instances (e.g. < 10) are statistically unreliable and are
marked with '*'.
"""

import argparse
import contextlib
import io
import json
import tempfile
from pathlib import Path

import yaml

from ultralytics import YOLO

from compute_size_ap import build_coco_gt, collect_predictions, list_images, resolve_labels_dir

SMALL_AREA = 32**2
MEDIUM_AREA = 96**2
MIN_RELIABLE = 10  # flag bins with fewer than this many GT instances


def run_coco_eval(gt, preds, cat_ids):
    """Run COCO evaluation restricted to ``cat_ids`` and return the stats list."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_path = tmp / "gt.json"
        pred_path = tmp / "pred.json"
        gt_path.write_text(json.dumps(gt))
        pred_path.write_text(json.dumps(preds))

        try:
            from faster_coco_eval import COCO, COCOeval_faster as COCOeval  # noqa: N813

            anno = COCO(str(gt_path))
            pred = anno.loadRes(str(pred_path))
            val = COCOeval(anno, pred, iouType="bbox", print_function=lambda *a, **k: None)
        except ImportError:
            from pycocotools.coco import COCO  # noqa: N813
            from pycocotools.cocoeval import COCOeval

            anno = COCO(str(gt_path))
            pred = anno.loadRes(str(pred_path))
            val = COCOeval(anno, pred, iouType="bbox")

        val.params.imgIds = [img["id"] for img in gt["images"]]
        val.params.catIds = list(cat_ids)
        val.evaluate()
        val.accumulate()
        with contextlib.redirect_stdout(io.StringIO()):
            val.summarize()
        return list(val.stats)


def fmt_ap(value, count, min_reliable=MIN_RELIABLE):
    """Format an AP value, appending '*' when the underlying bin has too few instances."""
    if count == 0:
        return "  -  "
    s = f"{value:.3f}"
    if count < min_reliable:
        s += "*"
    return s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to trained best.pt")
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size (default 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold (keep 0.001)")
    parser.add_argument("--device", default=None, help="device, e.g. 0 or cpu")
    args = parser.parse_args()

    data = yaml.safe_load(Path(args.data).read_text())
    data_dir = Path(args.data).parent
    val_src = data.get("val")
    if val_src is None:
        raise SystemExit("data.yaml has no 'val' key")
    images_dir = Path(val_src) if Path(val_src).is_absolute() else data_dir / val_src
    names = data.get("names", {})
    if isinstance(names, dict):
        names = {int(k): v for k, v in names.items()}
    nc = data.get("nc", len(names))

    images = list_images(images_dir)
    labels_dir = resolve_labels_dir(images_dir)
    gt = build_coco_gt(images, labels_dir, names)

    model = YOLO(args.model)
    preds = collect_predictions(model, images, args.imgsz, args.conf, args.device)
    print(f"val images: {len(images)}, predictions: {len(preds)}, GT instances: {len(gt['annotations'])}")

    # Overall row (all categories)
    overall = run_coco_eval(gt, preds, list(range(1, nc + 1)))

    header = (
        f"{'Class':<12} {'n':>5} {'AP':>8} {'AP50':>8} {'AP75':>8} "
        f"{'AP_small':>10} {'AP_medium':>11} {'AP_large':>10}"
    )
    print("\n" + "=" * len(header))
    print("Per-class x Per-size COCO AP  (AP @ IoU=0.50:0.95; * = n<{})".format(MIN_RELIABLE))
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    rows = []
    for cat_id in range(1, nc + 1):
        name = names.get(cat_id - 1, str(cat_id))
        anns = [a for a in gt["annotations"] if a["category_id"] == cat_id]
        n = len(anns)
        small_n = sum(1 for a in anns if a["area"] < SMALL_AREA)
        medium_n = sum(1 for a in anns if SMALL_AREA <= a["area"] < MEDIUM_AREA)
        large_n = sum(1 for a in anns if a["area"] >= MEDIUM_AREA)

        stats = run_coco_eval(gt, preds, [cat_id])
        rows.append((name, n, stats, small_n, medium_n, large_n))

        ap_small = fmt_ap(stats[3], small_n)
        ap_medium = fmt_ap(stats[4], medium_n)
        ap_large = fmt_ap(stats[5], large_n)
        print(
            f"{name:<12} {n:>5} {stats[0]:>8.3f} {stats[1]:>8.3f} {stats[2]:>8.3f} "
            f"{ap_small:>10} {ap_medium:>11} {ap_large:>10}"
        )

    print("-" * len(header))
    # Overall row
    all_small = sum(1 for a in gt["annotations"] if a["area"] < SMALL_AREA)
    all_medium = sum(1 for a in gt["annotations"] if SMALL_AREA <= a["area"] < MEDIUM_AREA)
    all_large = sum(1 for a in gt["annotations"] if a["area"] >= MEDIUM_AREA)
    print(
        f"{'ALL':<12} {len(gt['annotations']):>5} {overall[0]:>8.3f} {overall[1]:>8.3f} {overall[2]:>8.3f} "
        f"{fmt_ap(overall[3], all_small):>10} {fmt_ap(overall[4], all_medium):>11} {fmt_ap(overall[5], all_large):>10}"
    )
    print("=" * len(header))

    # Per-class size distribution
    print("\nPer-class GT size distribution:")
    for name, n, _, small_n, medium_n, large_n in rows:
        print(f"  {name:<12} n={n:>4}  small={small_n:>3}  medium={medium_n:>3}  large={large_n:>3}")


if __name__ == "__main__":
    main()
