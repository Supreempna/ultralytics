#!/usr/bin/env bash
# Breakthrough Experiments Runner for Cloud GPU Server
# Usage:
#   bash run_breakthrough.sh [epochs] [batch]
# Examples:
#   bash run_breakthrough.sh 20 16     # Fast screening mode (recommended first)
#   bash run_breakthrough.sh 50 16     # Full benchmark mode

set -e

EPOCHS=${1:-20}
BATCH=${2:-16}
DATA="ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml"
PROJECT="runs/breakthrough"

echo "=========================================================="
echo " Starting Breakthrough Experiments for Sonar Detection"
echo " Target Epochs: ${EPOCHS} | Batch Size: ${BATCH}"
echo "=========================================================="

# Track A: Wavelet Sub-band Disentanglement Denoising
echo "[1/3] Training Track A: YOLO26n-NAFNetWavelet..."
yolo train \
  model=ultralytics/cfg/models/26/yolo26n-NAFNetWavelet.yaml \
  data=${DATA} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  imgsz=640 \
  workers=8 \
  project=${PROJECT} \
  name=Wavelet_${EPOCHS}ep

# Evaluate Track A
python compute_class_size_ap.py \
  --model ${PROJECT}/Wavelet_${EPOCHS}ep/weights/best.pt \
  --data ${DATA} \
  --imgsz 640 > ${PROJECT}/Wavelet_${EPOCHS}ep/class_ap_report.txt 2>&1
cat ${PROJECT}/Wavelet_${EPOCHS}ep/class_ap_report.txt

# Track B: Acoustic Highlight-Shadow Dual Prior
echo "[2/3] Training Track B: YOLO26n-NAFNetAcoustic..."
yolo train \
  model=ultralytics/cfg/models/26/yolo26n-NAFNetAcoustic.yaml \
  data=${DATA} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  imgsz=640 \
  workers=8 \
  project=${PROJECT} \
  name=Acoustic_${EPOCHS}ep

# Evaluate Track B
python compute_class_size_ap.py \
  --model ${PROJECT}/Acoustic_${EPOCHS}ep/weights/best.pt \
  --data ${DATA} \
  --imgsz 640 > ${PROJECT}/Acoustic_${EPOCHS}ep/class_ap_report.txt 2>&1
cat ${PROJECT}/Acoustic_${EPOCHS}ep/class_ap_report.txt

# Track C: Hybrid Wavelet + Acoustic Prior
echo "[3/3] Training Track C: YOLO26n-NAFNetWaveletAcoustic..."
yolo train \
  model=ultralytics/cfg/models/26/yolo26n-NAFNetWaveletAcoustic.yaml \
  data=${DATA} \
  epochs=${EPOCHS} \
  batch=${BATCH} \
  imgsz=640 \
  workers=8 \
  project=${PROJECT} \
  name=WaveletAcoustic_${EPOCHS}ep

# Evaluate Track C
python compute_class_size_ap.py \
  --model ${PROJECT}/WaveletAcoustic_${EPOCHS}ep/weights/best.pt \
  --data ${DATA} \
  --imgsz 640 > ${PROJECT}/WaveletAcoustic_${EPOCHS}ep/class_ap_report.txt 2>&1
cat ${PROJECT}/WaveletAcoustic_${EPOCHS}ep/class_ap_report.txt

echo "=========================================================="
echo " All breakthrough experiments finished successfully!"
echo " Reports saved in ${PROJECT}/*/class_ap_report.txt"
echo "=========================================================="
