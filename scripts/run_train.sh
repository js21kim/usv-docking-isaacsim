#!/usr/bin/env bash
# 학습 실행 래퍼.
# ★ timeout 은 docker **클라이언트**만 죽이고 컨테이너는 계속 돈다.
#   실제로 잔여 컨테이너 2개가 GPU 를 나눠 써서 학습이 30배 느려진 적이 있다.
#   반드시 --name 을 붙이고 종료 시 docker kill 로 확실히 정리한다.
#
# ★ 로그 파일명에 타임스탬프를 붙인다. 예전에는 고정 이름을 쓰고 재시작 때마다
#   rm 으로 지웠는데, 그러면 시나리오 변경 이력을 잃는다.
#   체크포인트 폴더는 원래 타임스탬프라 안 겹쳤지만 텍스트 로그는 덮어써졌다.
set -u
NAME=$1; TASK=$2; ITERS=$3; ENVS=${4:-2048}
ROOT=/home/jason/07_USV_Docking_IsaacSim
docker kill "usvdock-$NAME" >/dev/null 2>&1 || true
docker run --rm --name "usvdock-$NAME" --gpus all --network host \
  -e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONUNBUFFERED=1 -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -v "$ROOT":/workspace/usvdock -w /workspace/usvdock \
  --entrypoint /isaac-sim/python.sh usvdock:latest \
  scripts/train.py --task "$TASK" --num_envs "$ENVS" --max_iterations "$ITERS" \
  --seed 1 --run_name "$NAME" --headless \
  > "$ROOT/install_logs/final_${NAME}_$(date +%y%m%d_%H%M%S).log" 2>&1
