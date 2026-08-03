#!/usr/bin/env bash
# BlueBoat 메시 내려받기.
#
# 저장소에 커밋하지 않는 이유:
#   ArduPilot SITL_Models 에서 온 자산이라 라이선스를 따로 따져야 하고,
#   13 MB 라 저장소를 무겁게 만든다. 원본에서 받는 편이 출처도 명확하다.
#
# 출처: https://github.com/ArduPilot/SITL_Models  Gazebo/models/blueboat
#       (원 CAD 는 Blue Robotics)
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/usvdock/assets/blueboat"
BASE="https://raw.githubusercontent.com/ArduPilot/SITL_Models/master/Gazebo/models/blueboat"
FILES=(
  meshes/blueboat_hull.dae meshes/blueboat_hull_collision.stl meshes/blueboat_hull_collision.dae
  meshes/blueboat_motor_port.dae meshes/blueboat_motor_stbd.dae
  meshes/blueboat_prop_port.dae meshes/blueboat_prop_stbd.dae
  meshes/blueboat_crosstube.dae meshes/blueboat_frame_asm_fore.dae meshes/blueboat_frame_asm_aft.dae
  model.sdf model.config
)
echo "BlueBoat 자산 → $DEST"
for f in "${FILES[@]}"; do
  mkdir -p "$DEST/$(dirname "$f")"
  if [ -s "$DEST/$f" ]; then printf '  skip  %s\n' "$f"; continue; fi
  curl -sfL "$BASE/$f" -o "$DEST/$f" && printf '  %6s %s\n' "$(du -h "$DEST/$f"|cut -f1)" "$f" \
    || { echo "  실패: $f" >&2; exit 1; }
done
echo "완료. model.sdf 에서 실측 물리 파라미터(모터 위치·관성·추력계수)를 읽는다."
