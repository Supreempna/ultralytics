#!/usr/bin/env python3
"""Compute COCO size-stratified AP (AP_small / AP_medium / AP_large) for a YOLO-format dataset.

What this does
--------------
1. Loads a trained checkpoint (best.pt) via ultralytics.
2. Reads a YOLO-format dataset's validation split (``images/val`` + ``labels/val``).
3. Runs inference and builds COCO-format ground-truth and prediction JSON (bbox = xywh,
   top-left origin, absolute pixels; area computed from original image size).
4. Evaluates with ``faster-coco-eval`` (falls back to ``pycocotools``) and reports:
   mAP(0.5:0.95), mAP50, mAP75, and AP split by object size.

COCO size thresholds (based on ground-truth bbox area in original-image pixels):
    small  : area <  32^2 = 1024 px^2
    medium : 32^2 <= area < 96^2 = 9216 px^2
    large  : area >= 96^2

Usage
-----
    python compute_size_ap.py --model runs/detect/train/weights/best.pt \
        --data ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml --imgsz 640

Notes
-----
- ``--conf 0.001`` (default) is required to build a full precision-recall curve.
- NMS IoU threshold is fixed at 0.7 (ultralytics validation default).
"""

import argparse
import json
import tempfile
from pathlib import Path

import cv2
import yaml

from ultralytics import YOLO

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def list_images(images_path):
    """Return a sorted list of image paths for a directory or a .txt image-list file."""
    images_path = Path(images_path)
    if images_path.is_dir():
        return sorted(p for p in images_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [Path(line.strip()) for line in images_path.read_text().splitlines() if line.strip()]


def resolve_labels_dir(images_dir):
    """Infer ``labels/<split>`` from ``images/<split>`` (standard YOLO layout)."""
    images_dir = Path(images_dir)
    return images_dir.parent.parent / "labels" / images_dir.name


def build_coco_gt(images, labels_dir, names):
    """Build COCO-format images + annotations from YOLO-format labels."""
    coco_images, coco_annotations = [], []
    ann_id = 0
    for img_id, img_path in enumerate(images, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warning] cannot read {img_path}, skipped")
            continue
        h, w = img.shape[:2]
        coco_images.append({"id": img_id, "file_name": img_path.name, "width": int(w), "height": int(h)})

        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            cx, cy, bw, bh = (float(x) for x in parts[1:5])
            # YOLO: normalized center-xywh -> COCO: absolute top-left xywh
            x = (cx - bw / 2.0) * w
            y = (cy - bh / 2.0) * h
            bw_abs, bh_abs = bw * w, bh * h
            coco_annotations.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO is 1-indexed
                    "bbox": [round(x, 3), round(y, 3), round(bw_abs, 3), round(bh_abs, 3)],
                    "area": round(bw_abs * bh_abs, 3),
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    categories = [{"id": i + 1, "name": names.get(i, str(i))} for i in range(len(names))]
    return {"images": coco_images, "annotations": coco_annotations, "categories": categories}


def collect_predictions(model, images, imgsz, conf, device):
    """Run inference over the images and return COCO-format predictions."""
    preds = []
    for img_id, img_path in enumerate(images, start=1):
        results = model.predict(str(img_path), imgsz=imgsz, conf=conf, iou=0.7, device=device, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xywh = boxes.xywh.cpu().numpy()  # center-xywh in original-image pixels
        cls = boxes.cls.cpu().numpy().astype(int)
        score = boxes.conf.cpu().numpy()
        for (xc, yc, w, h), c, s in zip(xywh, cls, score):
            preds.append(
                {
                    "image_id": img_id,
                    "category_id": int(c) + 1,
                    # convert center-xywh -> COCO top-left xywh
                    "bbox": [round(float(xc - w / 2), 3), round(float(yc - h / 2), 3), round(float(w), 3), round(float(h), 3)],
                    "score": round(float(s), 5),
                }
            )
    return preds


def coco_evaluate(gt, preds):
    """Run COCO evaluation and return (stats_list, n_annotations)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gt_path = tmp / "gt.json"
        pred_path = tmp / "pred.json"
        gt_path.write_text(json.dumps(gt))
        pred_path.write_text(json.dumps(preds))

        try:
            from faster_coco_eval import COCO, COCOeval_faster as COCOeval  # noqa: N813
            print("Evaluating with faster-coco-eval")
        except ImportError:
            from pycocotools.coco import COCO  # noqa: N813
            from pycocotools.cocoeval import COCOeval
            print("Evaluating with pycocotools")

        anno = COCO(str(gt_path))
        pred = anno.loadRes(str(pred_path))
        val = COCOeval(anno, pred, iouType="bbox")
        val.params.imgIds = [img["id"] for img in gt["images"]]
        val.evaluate()
        val.accumulate()
        val.summarize()
        return list(val.stats), len(gt["annotations"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to trained best.pt")
    parser.add_argument("--data", required=True, help="path to data.yaml")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size (default 640)")
    parser.add_argument("--conf", type=float, default=0.001, help="confidence threshold (keep 0.001 for full PR curve)")
    parser.add_argument("--device", default=None, help="device, e.g. 0, 0,1, or cpu")
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

    images = list_images(images_dir)
    labels_dir = resolve_labels_dir(images_dir)
    print(f"val images: {len(images)}")
    print(f"labels dir: {labels_dir}")

    gt = build_coco_gt(images, labels_dir, names)

    model = YOLO(args.model)
    preds = collect_predictions(model, images, args.imgsz, args.conf, args.device)
    print(f"predictions: {len(preds)}")

    stats, n_gt = coco_evaluate(gt, preds)

    small = sum(1 for a in gt["annotations"] if a["area"] < 32**2)
    medium = sum(1 for a in gt["annotations"] if 32**2 <= a["area"] < 96**2)
    large = sum(1 for a in gt["annotations"] if a["area"] >= 96**2)

    print("\n" + "=" * 56)
    print("COCO size-stratified AP  (AP @ IoU=0.50:0.95)")
    print("=" * 56)
    print(f"GT instances        : {n_gt}  (small={small}, medium={medium}, large={large})")
    print(f"mAP (0.5:0.95)      : {stats[0]:.4f}")
    print(f"mAP50               : {stats[1]:.4f}")
    print(f"mAP75               : {stats[2]:.4f}")
    print("-" * 56)
    print(f"AP_small  (< 32^2)  : {stats[3]:.4f}")
    print(f"AP_medium (32^2-96^2): {stats[4]:.4f}")
    print(f"AP_large  (>= 96^2) : {stats[5]:.4f}")
    print("=" * 56)


if __name__ == "__main__":
    main()
