#!/usr/bin/env bash
# Breakthrough Experiments Runner for Cloud GPU Server (Robust Version)
# Usage:
#   bash run_breakthrough.sh [track] [epochs] [batch]
# Examples:
#   bash run_breakthrough.sh all 20 16       # Run all tracks sequentially (20 epochs)
#   bash run_breakthrough.sh Acoustic 20 16  # Run only Track B (Acoustic)
#   bash run_breakthrough.sh WaveletAcoustic 20 16 # Run only Track C
#   bash run_breakthrough.sh Wavelet 50 16   # Run Wavelet 50 epochs

TRACK=${1:-all}
EPOCHS=${2:-20}
BATCH=${3:-16}
DATA="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml"
PROJECT="breakthrough"  # ultralytics defaults to runs/detect/$PROJECT/<name>

# Do NOT use set -e so that one evaluation error doesn't kill the entire training queue!
set +e

echo "=========================================================="
echo " Starting Breakthrough Experiments for Sonar Detection"
echo " Target Track: ${TRACK} | Epochs: ${EPOCHS} | Batch Size: ${BATCH}"
echo "=========================================================="

# Check if pycocotools is available for class-size evaluation
python -c "import pycocotools" 2>/dev/null || python -c "import faster_coco_eval" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[INFO] Installing pycocotools for class-size AP evaluation..."
    pip install pycocotools -q || echo "[WARN] pip install pycocotools failed; class AP eval may be skipped."
fi

run_single_track() {
    local model_yaml=$1
    local exp_name=$2
    local label=$3

    echo ""
    echo "=========================================================="
    echo "[$(date)] Starting ${label}: ${exp_name}"
    echo "  Model: ${model_yaml}"
    echo "=========================================================="

    # Run YOLO training
    # Check if yolo or /opt/conda/bin/yolo
    if command -v yolo &> /dev/null; then
        YOLO_CMD="yolo"
    elif [ -f "/opt/conda/bin/yolo" ]; then
        YOLO_CMD="/opt/conda/bin/yolo"
    else
        YOLO_CMD="python -m ultralytics"
    fi

    $YOLO_CMD train \
      model="${model_yaml}" \
      data="${DATA}" \
      epochs="${EPOCHS}" \
      batch="${BATCH}" \
      imgsz=640 \
      workers=8 \
      project="${PROJECT}" \
      name="${exp_name}"

    echo "[$(date)] Training finished for ${exp_name}"

    # Locate best.pt robustly (Ultralytics places it under runs/detect/$PROJECT/$exp_name or $PROJECT/$exp_name)
    local best_pt=""
    if [ -f "runs/detect/${PROJECT}/${exp_name}/weights/best.pt" ]; then
        best_pt="runs/detect/${PROJECT}/${exp_name}/weights/best.pt"
    elif [ -f "${PROJECT}/${exp_name}/weights/best.pt" ]; then
        best_pt="${PROJECT}/${exp_name}/weights/best.pt"
    elif [ -f "runs/${PROJECT}/${exp_name}/weights/best.pt" ]; then
        best_pt="runs/${PROJECT}/${exp_name}/weights/best.pt"
    fi

    if [ -n "$best_pt" ] && [ -f "$best_pt" ]; then
        echo "[INFO] Found checkpoint: ${best_pt}"
        local report_dir="$(dirname $(dirname ${best_pt}))"
        echo "[INFO] Running per-class evaluation via compute_class_size_ap.py..."
        python compute_class_size_ap.py --model "${best_pt}" --data "${DATA}" --imgsz 640 > "${report_dir}/class_ap_report.txt" 2>&1 || true
        if [ -f "${report_dir}/class_ap_report.txt" ]; then
            cat "${report_dir}/class_ap_report.txt"
        fi
    else
        echo "[WARN] Could not find best.pt for ${exp_name}; skipping class AP report."
    fi
}

# Track A: Wavelet
if [ "$TRACK" = "all" ] || [ "$TRACK" = "Wavelet" ]; then
    run_single_track "ultralytics/cfg/models/26/yolo26n-NAFNetWavelet.yaml" "Wavelet_${EPOCHS}ep" "Track A (Wavelet)"
fi

# Track B: Acoustic
if [ "$TRACK" = "all" ] || [ "$TRACK" = "Acoustic" ]; then
    run_single_track "ultralytics/cfg/models/26/yolo26n-NAFNetAcoustic.yaml" "Acoustic_${EPOCHS}ep" "Track B (Acoustic)"
fi

# Track C: WaveletAcoustic
if [ "$TRACK" = "all" ] || [ "$TRACK" = "WaveletAcoustic" ]; then
    run_single_track "ultralytics/cfg/models/26/yolo26n-NAFNetWaveletAcoustic.yaml" "WaveletAcoustic_${EPOCHS}ep" "Track C (WaveletAcoustic)"
fi

echo ""
echo "=========================================================="
echo " All requested experiments finished!"
echo "=========================================================="
