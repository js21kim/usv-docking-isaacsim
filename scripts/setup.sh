#!/usr/bin/env bash
# 신규 머신 설치 — Isaac Sim 부터 usvdock 이미지까지.
#
# 이미지 체인:
#   nvcr.io/nvidia/isaac-sim:5.1.0
#     └ isaac-lab-base          IsaacLab 포크의 docker/ 구성으로 빌드
#         └ isaac-lab-base-fixed  ← 누락된 isaaclab 코어 복구 (아래 설명)
#             └ usvdock
#
# ★ isaac-lab-base 는 그냥 빌드하면 **반쪽짜리로 나온다**:
#   isaaclab==0.54.0 의 의존성 flatdict==4.0.1 이 sdist 이고 구식 setup.py 를 쓰는데,
#   pip 이 격리 빌드환경에 받는 최신 setuptools 에는 pkg_resources 가 없어 실패한다.
#   isaaclab.sh --install 은 실패 모듈을 건너뛰므로 **docker build 는 exit 0** 이고,
#   핵심 isaaclab 만 빠진 이미지가 나온다(나머지 5개 확장은 정상).
#   docker/Dockerfile.isaaclab-fix 가 그 복구 레이어다.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="${WORK:-$HOME/isaaclab_ws}"

say(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }
die(){ printf '\n\033[1;31m[중단] %s\033[0m\n' "$*" >&2; exit 1; }

say "사전 점검"
docker info >/dev/null 2>&1 || die "docker 를 사용할 수 없습니다. 재로그인하거나 docker 그룹을 확인하세요."
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1 \
  || die "GPU 컨테이너 실행 실패. nvidia-container-toolkit 을 확인하세요."
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
echo "  드라이버 $DRV  (Isaac Sim 5.1 은 580 이상 필요)"
[ "${DRV%%.*}" -ge 580 ] || die "드라이버 580 이상이 필요합니다."

say "BlueBoat 자산"
bash "$ROOT/scripts/fetch_assets.sh"

say "IsaacLab 포크 클론 → $WORK/isaaclab"
mkdir -p "$WORK"
if [ -d "$WORK/isaaclab/.git" ]; then git -C "$WORK/isaaclab" fetch --all --prune
else git clone https://github.com/luckkim123/IsaacLab.git "$WORK/isaaclab"; fi

say "isaac-lab-base 빌드"
# container.py 는 X11 여부를 대화형으로 묻는다. 비대화 실행에서 EOFError 로 죽으므로
# 답을 미리 파일에 박아 둔다. (0 = 비활성. GUI 가 필요하면 1 로 바꾸고 재시작)
printf '[X11]\nx11_forwarding_enabled = 0\n\n' > "$WORK/isaaclab/docker/.container.cfg"
( cd "$WORK/isaaclab/docker" && python3 container.py start base < /dev/null )
docker stop isaac-lab-base >/dev/null 2>&1 || true

say "isaac-lab-base-fixed 빌드 (isaaclab 코어 복구)"
docker build -f "$ROOT/docker/Dockerfile.isaaclab-fix" -t isaac-lab-base-fixed:latest "$ROOT/docker"

say "usvdock 빌드"
docker build -f "$ROOT/docker/Dockerfile.usvdock" -t usvdock:latest \
  --build-context docker="$ROOT/docker" "$ROOT"

say "검증"
docker run --rm --gpus all -e OMNI_KIT_ALLOW_ROOT=1 -e PYTHONUNBUFFERED=1 \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  --entrypoint /isaac-sim/python.sh usvdock:latest \
  -c "import isaaclab, usvdock, gymnasium as gym; \
      print('isaaclab', isaaclab.__version__); \
      print('tasks:', sorted(k for k in gym.envs.registry if 'USVDock' in k))"

cat <<'MSG'

  ─────────────────────────────────────────────────────────────
   설치 완료.  다음 단계:

     bash scripts/run_train.sh test Isaac-USVDock-M1-Lidar-v0 50 512
     python3 scripts/test_geometry.py        # Isaac Sim 불필요, 14개 단위시험

   실행 규칙은 README.md 의 "반드시 지킬 것" 을 볼 것.
  ─────────────────────────────────────────────────────────────
MSG
