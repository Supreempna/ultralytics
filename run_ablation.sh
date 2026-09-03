#!/usr/bin/env bash
# NAFNet 容量消融：顺序跑 3 个轻量配置（20 轮粗筛）
# 每个配置输出到 runs/ablation/<name>/
set -e

DATA=ultralytics/dataset/FLS_Detection_YOLO/FLSD.yaml
PROJECT=ablation
EPOCHS=20
IMGSZ=640
BATCH=16

run() {
  local yaml=$1
  local name=$2
  echo ""
  echo "=============================================================="
  echo "[$(date)] 开始训练: $name  ($yaml)"
  echo "=============================================================="
  /opt/conda/bin/yolo train \
    model="$yaml" data="$DATA" \
    epochs="$EPOCHS" imgsz="$IMGSZ" batch="$BATCH" \
    project="$PROJECT" name="$name"
  echo "[$(date)] $name 完成"
}

# ---- 3 个容量消融配置（顺序执行）----
run ultralytics/cfg/models/26/yolo26n-NAFNet-m16b1.yaml m16b1
run ultralytics/cfg/models/26/yolo26n-NAFNet-m8b2.yaml  m8b2
run ultralytics/cfg/models/26/yolo26n-NAFNet-m8b1.yaml  m8b1

# ---- 可选：完整块 NAFNetFull（Token Mixing + FFN），显存更高，谨慎开 ----
# 若 OOM，把 batch 改成 8
# run ultralytics/cfg/models/26/yolo26n-NAFNetFull.yaml nafnetfull

echo ""
echo "=============================================================="
echo "全部完成，汇总最终 mAP50-95："
echo "=============================================================="
for name in m16b1 m8b2 m8b1; do
  f="$PROJECT/$name/results.csv"
  if [ -f "$f" ]; then
    map=$(tail -n 1 "$f" | awk -F',' '{print $9}')
    echo "  $name  : mAP50-95 = $map"
  else
    echo "  $name  : (未找到 results.csv)"
  fi
done
echo "结果文件: $PROJECT/{m16b1,m8b2,m8b1}/results.csv"
