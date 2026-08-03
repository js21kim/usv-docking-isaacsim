"""지오메트리·센서 모듈 검증 (Isaac Sim 불필요).

여기서 잡아야 할 것:
  - LiDAR 가 핑거를 보는가, 그리고 수중 기초부를 **못 보는가**
  - 마스트 높이가 종단 가시성을 좌우하는가 (Mid-360 수직시야 -7°)
  - FLS 가 기초부를 보는가, 미션1에서는 **안 보이는가**
  - 목표 자세가 기초부에 따라 얕아지는가
  - 프로펠러 충돌 판정이 동작하는가
"""

import math
import sys

import torch

sys.path.insert(0, "/home/jason/07_USV_Docking_IsaacSim")
from usvdock import blueboat_cfg as C
from usvdock import geometry as G

dev = "cpu"
OK, FAIL = "\033[32mOK\033[0m", "\033[31mFAIL\033[0m"
results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"  [{OK if cond else FAIL}] {name}" + (f"  — {detail}" if detail else ""))


def make(n=1, mission=2, **over):
    p = G.sample_scene(n, dev, mission=mission, current_ms=0.0)
    for k, v in over.items():
        setattr(p, k, torch.full((n,), float(v), device=dev))
    return p


yaw = torch.tensor([math.pi / 2], device=dev)  # 벽(+y)을 향해
p = make(mission=2, berth_x=0.0, footing_protrude=2.0, footing_top_depth=0.20,
         footing_width=3.0, footing_x_offset=0.0)

print("=" * 72)
print("1. LiDAR — 핑거는 보이고, 수중 기초부는 안 보여야 한다")
print("=" * 72)
pos = torch.tensor([[0.0, G.WALL_Y - 6.0]], device=dev)
scan = G.lidar_scan(pos, yaw, p)
n_hit = int((scan < 0.999).sum())
check("전방에서 검출됨", n_hit > 0, f"{n_hit}/{scan.shape[1]} 빔 반사")
d_fwd = float(scan[0, 0]) * C.SensorMountCfg().lidar_obs_max_range
check("선수 방향 = 벽까지 6 m (수중 기초부 2 m 는 무시)", 5.5 < d_fwd < 6.5, f"{d_fwd:.2f} m")

print()
print("=" * 72)
print("2. Mid-360 수직시야 -7° — 마스트 높이가 종단 가시성을 좌우하는가")
print("=" * 72)
print("   핑거 상단 0.50 m. 마스트를 그보다 높이 달면 종단에서 핑거가 사라져야 한다.")
print("   ※ 배는 핑거 바깥(y < 7.5)에 둔다. 안쪽은 이미 충돌 상황이라 센싱 시험이 아니다.")
print("   빔 개수로는 판별할 수 없다 — 먼 벽에 항상 맞아 72개로 포화된다.")
print("   **최소 거리**를 본다. 핑거가 보이면 gap, 안 보이면 벽까지 (2.5+gap) 로 뛴다.")
gaps = (3.0, 1.5, 0.6, 0.2)
R = C.SensorMountCfg().lidar_obs_max_range
rows = {}
# 핑거 안쪽면 x=±0.8, 근단 y=7.5 → 최소거리는 **모서리까지** sqrt(0.8² + gap²)
expect = [math.hypot(0.8, g) for g in gaps]
print(f"    핑거 모서리까지 기대거리 = {['%.2f' % e for e in expect]}")
for mast in (0.45, 0.70, 1.20):
    m = C.SensorMountCfg()
    m.lidar_pos = (0.0, 0.0, mast)   # 인스턴스에 직접 설정해야 반영된다
    row = []
    for gap in gaps:
        pp = torch.tensor([[0.0, G.WALL_Y - 2.5 - gap]], device=dev)
        row.append(float(G.lidar_scan(pp, yaw, p, mount=m).min()) * R)
    rows[mast] = row
    txt = "  ".join(f"{v:5.2f}" for v in row)
    print(f"    마스트 {mast:.2f} m → 최소거리 = {txt}")
low_ok = all(abs(rows[0.45][i] - e) < 0.15 for i, e in enumerate(expect))
high_lost = rows[1.20][-1] > expect[-1] + 0.5
check("낮은 마스트(0.45)는 종단까지 핑거 추적", low_ok, f"{rows[0.45]}")
check("높은 마스트(1.20)는 종단에서 핑거 상실", high_lost,
      f"gap 0.2 m 에서 {rows[1.20][-1]:.2f} m — 벽으로 튐. 마스트를 낮게 단 설계 근거")

print()
print("=" * 72)
print("3. FLS — 기초부를 보는가 / 미션1에서는 안 보이는가")
print("=" * 72)
print("   min() 이 아니라 **중앙 빔**을 본다. 핑거 수중부가 더 가까워 min 은 핑거를 집는다.")
pos = torch.tensor([[0.0, G.WALL_Y - 4.0]], device=dev)
p2 = make(mission=2, berth_x=0.0, footing_protrude=2.0, footing_top_depth=0.20,
          footing_width=3.0, footing_x_offset=0.0)
p1 = make(mission=1, berth_x=0.0, footing_protrude=2.0, footing_top_depth=0.20,
          footing_width=3.0, footing_x_offset=0.0)
mid = C.SensorMountCfg().fls_n_beams // 2
R = C.SensorMountCfg().fls_max_range
d2 = float(G.fls_scan(pos, yaw, p2)[0, mid]) * R
d1 = float(G.fls_scan(pos, yaw, p1)[0, mid]) * R
print(f"    중앙빔  미션2 {d2:.2f} m (기초부 예상 2.0)   미션1 {d1:.2f} m (벽 예상 4.0)")
check("미션2 중앙빔 = 기초부까지 2 m", abs(d2 - 2.0) < 0.35, f"{d2:.2f} m")
check("미션1 중앙빔 = 벽까지 4 m", abs(d1 - 4.0) < 0.35, f"{d1:.2f} m")
check("FLS 가 두 미션을 구분", d2 < d1 - 1.0, f"{d1-d2:.2f} m 차이")

print()
print("=" * 72)
print("4. 목표 자세 — 기초부가 위험하면 얕아지는가")
print("=" * 72)
safe = make(mission=2, berth_x=0.0, footing_protrude=2.0, footing_top_depth=0.32)
risk = make(mission=2, berth_x=0.0, footing_protrude=2.0, footing_top_depth=0.18)
ts, tr = G.docking_target(safe), G.docking_target(risk)
print(f"    기초부 수심 0.32 m (프롭 0.272 보다 깊음, 무해) → 목표 y = {float(ts[0,1]):.3f}")
print(f"    기초부 수심 0.18 m (프롭보다 얕음,   위험) → 목표 y = {float(tr[0,1]):.3f}")
check("위험 시 목표가 후퇴", float(tr[0, 1]) < float(ts[0, 1]) - 0.1,
      f"{float(ts[0,1])-float(tr[0,1]):.3f} m 후퇴")

print()
print("=" * 72)
print("5. 프로펠러 충돌 판정 (선체가 아니라 프로펠러 기준)")
print("=" * 72)
b = C.BerthCfg()
deep = torch.tensor([[0.0, G.WALL_Y - 1.0]], device=dev)
prop_y = float(deep[0, 1]) - b.prop_aft_offset
edge = G.WALL_Y - float(risk.footing_protrude[0])
col = G.check_collision(deep, yaw, risk)
print(f"    프로펠러 y={prop_y:.2f}, 기초부 안쪽끝 y={edge:.2f} → "
      f"{'기초부 위' if prop_y > edge else '기초부 바깥'}")
check("얕은 기초부 위 → 충돌", bool(col["prop_strike"][0]) == (prop_y > edge))
check("깊은 기초부 위 → 충돌 없음", not bool(G.check_collision(deep, yaw, safe)["prop_strike"][0]))

print()
print("=" * 72)
print("6. 배치 동작 (1024 env) + 위험 발생률")
print("=" * 72)
N = 1024
pb = G.sample_scene(N, dev, mission=2, current_ms=(0.0, 1.75))
posb = torch.stack([pb.berth_x, torch.full((N,), G.WALL_Y - 5.0, device=dev)], dim=-1)
yawb = torch.full((N,), math.pi / 2, device=dev)
lb, fb, tb = G.lidar_scan(posb, yawb, pb), G.fls_scan(posb, yawb, pb), G.docking_target(pb)
check("LiDAR 형상", tuple(lb.shape) == (N, 72), str(tuple(lb.shape)))
check("FLS 형상", tuple(fb.shape) == (N, 128), str(tuple(fb.shape)))
check("NaN 없음", bool(torch.isfinite(lb).all() and torch.isfinite(fb).all()))
nom_y = G.WALL_Y - 0.30 - C.LOA / 2
retreat = (tb[:, 1] < nom_y - 1e-4).float().mean()
print(f"    프롭보다 얕은 기초부 : {100*float((pb.footing_top_depth < b.prop_depth).float().mean()):.1f}%")
print(f"    목표 후퇴 에피소드   : {100*float(retreat):.1f}%")
check("후퇴 비율이 유의미", 0.15 < float(retreat) < 0.85)

print()
print("=" * 72)
print(f"결과: {sum(results)}/{len(results)} 통과")
print("=" * 72)
sys.exit(0 if all(results) else 1)
