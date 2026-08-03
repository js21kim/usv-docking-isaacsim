#!/usr/bin/env bash
# 본 실험 4회 — 폐어구 시나리오. GPU 경합을 피해 순차 실행.
#   1단계(쉬움, 폐어구 없음):  B_M1 = RL(LiDAR)
#   2단계(어려움, 폐어구):     C_M2 = RL(LiDAR+FLS)
#   2단계 절제:                B_M2 = RL(LiDAR only)  ← 폐어구에 돌진할 것
#   통제군:                    C_M1 = RL(융합, 폐어구 없음) → B_M1 과 같아야 정상
set -u
cd /home/jason/07_USV_Docking_IsaacSim
ITERS=${ITERS:-3000}
for spec in "B_M1 Isaac-USVDock-M1-Lidar-v0" "C_M2 Isaac-USVDock-M2-Fusion-v0" \
            "B_M2 Isaac-USVDock-M2-Lidar-v0" "C_M1 Isaac-USVDock-M1-Fusion-v0"; do
  set -- $spec
  echo "=== $1 시작 $(date +%H:%M:%S) ==="
  bash scripts/run_train.sh "$1" "$2" "$ITERS" 2048
  echo "=== $1 종료 $(date +%H:%M:%S) ==="
  grep -E "success_rate|entangle_rate|finger_hit_rate" "$(ls -t install_logs/final_$1_*.log|head -1)" | tail -3
done
echo "=== 전체 완료 $(date +%H:%M:%S) ==="
