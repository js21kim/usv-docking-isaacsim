"""장면 지오메트리와 해석적 레이캐스팅.

Isaac Lab 의 RayCaster(메시 등록 방식) 대신 해석적 교차 계산을 쓰는 이유:

  (1) 장면 구성물이 전부 **축 정렬 박스(AABB)** 다 — 벽, 핑거, 수중 기초부.
      슬랩(slab) 기법으로 정확히 풀린다. 근사가 아니다.
  (2) **어떤 센서가 무엇을 보는지 정확히 통제**해야 한다.
      LiDAR 는 수면 위 고체만, FLS 는 수중 물체만 본다.
      메시 기반 RayCaster 에 수면을 넣으면 광선이 수면에 맞아 실제 Mid-360 이
      결코 주지 않는 거리값을 돌려준다 (blueboat_cfg.SensorMountCfg 주석 참조).
      대상 집합을 센서별로 나누는 편이 안전하고 명시적이다.
  (3) 에피소드마다 기초부 형상을 무작위화해야 하는데, 파라미터 텐서로 다루면
      USD 프림을 다시 만들 필요가 없다.

좌표계:
    x  수조 길이 방향 (35 m), 중앙 0  → [-17.5, +17.5].  **유속이 이 축을 따라 흐른다**
    y  수조 폭   방향 (20 m), 중앙 0  → [-10, +10].      버스는 y=+10 긴 벽에 있다
    z  상방, 수면 z=0, 바닥 z=-9.6
    yaw  +x 축 기준 반시계

버스 배치 (사용자 설계):
    벽 y=+10 에 핑거(막대) 2개가 수직으로 돌출. 그 사이에 선수부터 진입.
    핑거는 수면 위 0.5 m 까지 올라와 LiDAR 에 보인다.
    미션2 의 수중 기초부는 벽 아래에서 안쪽으로 돌출하며 LiDAR 에 원리적으로 안 보인다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from . import blueboat_cfg as C

WALL_Y = 10.0  # 버스가 있는 긴 벽의 y 좌표
# 핑거 근단(슬롯 입구) y. 보상 설계에서 "진입 임박도"를 재는 기준선이다.
BerthCfg_pile_len = C.BerthCfg().pile_length
_EPS = 1e-6


# =============================================================================
# 에피소드별 무작위 장면 파라미터
# =============================================================================
MAX_OBSTACLES = 3  # 에피소드당 최대 폐어구 개수


@dataclass
class SceneParams:
    """환경마다 다른 장면 파라미터. 전부 (N,...) 텐서.

    무작위화가 없으면 실험이 무너진다: 폐어구를 고정하면 LiDAR-only 정책이
    보이지도 않는 장애물을 위치로 외워서 피해버리고, FLS 의 효과가 사라진다.
    """

    berth_x: torch.Tensor  # (N,) 버스 중심 x
    current_u: torch.Tensor  # (N,) 유속 x 성분 [m/s]
    has_gear: torch.Tensor  # (N,) bool — 미션1은 False, 미션2는 True

    # --- 폐어구(유령어구) ---
    # 뜸줄에 매달려 상단이 수면 바로 아래에 있고, 발줄·그물이 아래로 늘어진다.
    #   → 수면 위로 안 나오므로 LiDAR 에 **원리적으로** 안 보인다
    #   → 수중으로 늘어져 있으므로 FLS 에는 보인다
    #   → 상단이 프로펠러 최심점(0.272 m)보다 항상 얕다 → **검출 = 무조건 위험**
    # 이 마지막 성질이 핵심이다. FLS 는 고도각이 모호해 깊이를 못 재는데,
    # 폐어구는 언제나 위험하므로 깊이를 몰라도 판단이 완성된다.
    # (연승줄을 탈락시킨 이유: 위험 여부가 교차점 깊이로 갈려 FLS 로 판단 불가)
    gear_x: torch.Tensor  # (N,K)
    gear_y: torch.Tensor  # (N,K)
    gear_w: torch.Tensor  # (N,K) x 방향 폭
    gear_l: torch.Tensor  # (N,K) y 방향 길이
    gear_yaw: torch.Tensor  # (N,K) 평면상 방향 [rad]
    gear_top: torch.Tensor  # (N,K) 상단 수심 (양수)
    gear_bot: torch.Tensor  # (N,K) 하단 수심 (양수)
    gear_on: torch.Tensor  # (N,K) bool — 활성 여부

    def index(self, idx: torch.Tensor) -> "SceneParams":
        return SceneParams(**{k: getattr(self, k)[idx] for k in self.__dataclass_fields__})


def sample_scene(
    num_envs: int,
    device: str,
    mission: int,
    current_ms: float | tuple[float, float] = 0.0,
    generator: torch.Generator | None = None,
) -> SceneParams:
    """에피소드 장면을 샘플링한다."""

    def u(lo, hi, *shape):
        sh = shape or (num_envs,)
        return torch.rand(*sh, device=device, generator=generator) * (hi - lo) + lo

    b = C.BerthCfg()
    t = C.TankCfg()
    K = MAX_OBSTACLES

    berth_x = u(-t.length / 2 + 6.0, t.length / 2 - 6.0)

    if isinstance(current_ms, tuple):
        cur = u(*current_ms)
    else:
        cur = torch.full((num_envs,), float(current_ms), device=device)
    sign = torch.where(torch.rand(num_envs, device=device, generator=generator) < 0.5, -1.0, 1.0)

    has_gear = torch.full((num_envs,), mission == 2, device=device, dtype=torch.bool)

    # --- 폐어구 배치 ---
    # 접근 회랑에만 둔다. 슬롯 입구(y=7.5)에서 1 m 이상, 선박 초기위치(y<=2.6)보다 안쪽.
    #   → y ∈ [3.0, 6.5].  이 범위 밖이면 우회 불가 에피소드가 생겨 지표가 오염된다.
    # 배치 폭 ±5.0 m 는 실측으로 정했다 — 직선 접근이 폐어구에 걸리는 비율:
    #   ±3.5 m → 64.5%   ±5.0 m → 51.4%
    # 약 절반의 에피소드가 회피를 요구한다. 이보다 높으면 도킹보다 회피가 주가 되고,
    # 낮으면 FLS 유무의 차이가 묻힌다.
    gx = berth_x.view(-1, 1) + u(-5.0, 5.0, num_envs, K)
    gy = u(WALL_Y - 7.0, WALL_Y - 3.5, num_envs, K)
    gw = u(*b.gear_size_range, num_envs, K)
    gl = u(*b.gear_size_range, num_envs, K)
    gyaw = torch.rand(num_envs, K, device=device, generator=generator) * math.pi
    gtop = u(*b.gear_top_depth_range, num_envs, K)
    gbot = gtop + u(*b.gear_hang_range, num_envs, K)

    # 개수 1~K 무작위: 앞에서부터 n 개만 활성
    # 개수 1~3. 크기 0.5~1.0 m + 여유 0.5 m 조합에서 우회로 존재율 98.1% 로 확인.
    n_on = torch.randint(1, K + 1, (num_envs,), device=device, generator=generator)
    idx = torch.arange(K, device=device).view(1, K)
    gear_on = (idx < n_on.view(-1, 1)) & has_gear.view(-1, 1)

    return SceneParams(
        berth_x=berth_x,
        current_u=cur * sign,
        has_gear=has_gear,
        gear_x=gx, gear_y=gy, gear_w=gw, gear_l=gl, gear_yaw=gyaw,
        gear_top=gtop, gear_bot=gbot, gear_on=gear_on,
    )


# =============================================================================
# 박스 목록 만들기
# =============================================================================
def _finger_boxes(p: SceneParams) -> tuple[torch.Tensor, torch.Tensor]:
    """핑거 2개의 xy AABB. 반환 (N, 2, 4) = [x_min, x_max, y_min, y_max]와 z범위 (N,2,2)."""
    b = C.BerthCfg()
    half_w = b.pile_diameter / 2
    off = b.pile_gap / 2 + half_w  # 중심에서 핑거 중심까지
    y_min = WALL_Y - b.pile_length
    y_max = WALL_Y

    cx = torch.stack([p.berth_x - off, p.berth_x + off], dim=-1)  # (N,2)
    xy = torch.stack(
        [cx - half_w, cx + half_w, torch.full_like(cx, y_min), torch.full_like(cx, y_max)], dim=-1
    )  # (N,2,4)
    # z 범위: **수상 구조물**. 수면 아래로 0.05 m 만 잠긴다.
    #   FLS 에 핑거 반사를 거의 남기지 않아 "LiDAR=수면 위 / FLS=수면 아래" 분업이 선명해진다.
    #   완전 분리는 아니다 — 안벽 자체는 수면 아래로 이어지므로 FLS 에 잡힌다.
    #   그것이 오히려 상보성의 좋은 예시다(같은 구조물의 다른 부분을 나눠 본다).
    z = torch.tensor([-0.05, b.pile_height_above_water], device=p.berth_x.device)
    z = z.view(1, 1, 2).expand(cx.shape[0], 2, 2)
    return xy, z


def _gear_obb(p: SceneParams) -> tuple[torch.Tensor, torch.Tensor]:
    """폐어구를 **방향 있는 직육면체(OBB)** 로 반환.

    반환:
        obb: (N,K,5) = [cx, cy, half_w, half_l, yaw]
        z:   (N,K,2) = [z_bot, z_top]  (수직축은 고정 — 아래로 늘어진 형상)

    비활성(미션1 또는 개수 미달)은 장면 밖으로 치운다.
    사후에 히트를 무효화하면 광선이 거기서 멈춰 뒤쪽 물체를 못 보게 된다.
    """
    obb = torch.stack(
        [p.gear_x, p.gear_y, p.gear_w / 2, p.gear_l / 2, p.gear_yaw], dim=-1
    )
    far = torch.tensor([1e6, 1e6, 0.5, 0.5, 0.0], device=p.gear_x.device).view(1, 1, 5)
    obb = torch.where(p.gear_on.unsqueeze(-1), obb, far)
    z = torch.stack([-p.gear_bot, -p.gear_top], dim=-1)
    return obb, z


def _ray_obb_2d(
    origin: torch.Tensor,  # (N,2)
    dirs: torch.Tensor,  # (N,R,2)
    obb: torch.Tensor,  # (N,K,5) [cx,cy,hw,hl,yaw]
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """광선 vs 방향 있는 직사각형. 광선을 상자 국소좌표로 회전시켜 slab 기법을 그대로 쓴다.

    AABB 판정과 수학은 같다 — 회전만 먼저 하면 된다.
    """
    N, R, _ = dirs.shape
    K = obb.shape[1]
    cx, cy = obb[..., 0], obb[..., 1]
    hw, hl = obb[..., 2], obb[..., 3]
    yaw = obb[..., 4]
    c, sn = torch.cos(yaw), torch.sin(yaw)  # (N,K)

    # 원점을 상자 국소좌표로
    ox = origin[:, 0:1] - cx  # (N,K)
    oy = origin[:, 1:2] - cy
    lox = (ox * c + oy * sn).view(N, 1, K)
    loy = (-ox * sn + oy * c).view(N, 1, K)

    # 방향도 회전 (평행이동 없음)
    dx, dy = dirs[..., 0].unsqueeze(-1), dirs[..., 1].unsqueeze(-1)  # (N,R,1)
    cc, ss = c.view(N, 1, K), sn.view(N, 1, K)
    ldx = dx * cc + dy * ss  # (N,R,K)
    ldy = -dx * ss + dy * cc

    inv_x = 1.0 / torch.where(ldx.abs() < _EPS, torch.full_like(ldx, _EPS), ldx)
    inv_y = 1.0 / torch.where(ldy.abs() < _EPS, torch.full_like(ldy, _EPS), ldy)
    HW, HL = hw.view(N, 1, K), hl.view(N, 1, K)

    t1x, t2x = (-HW - lox) * inv_x, (HW - lox) * inv_x
    t1y, t2y = (-HL - loy) * inv_y, (HL - loy) * inv_y
    t_near = torch.maximum(torch.minimum(t1x, t2x), torch.minimum(t1y, t2y))
    t_far = torch.minimum(torch.maximum(t1x, t2x), torch.maximum(t1y, t2y))

    valid = (t_far >= t_near) & (t_far > 0)
    t_enter = torch.where(t_near > 0, t_near, torch.zeros_like(t_near))
    dist = torch.where(valid, t_enter, torch.full_like(t_enter, float("inf")))
    dist = torch.where(dist <= max_range, dist, torch.full_like(dist, float("inf")))

    best, idx = dist.min(dim=-1)
    hit = torch.isfinite(best)
    return torch.where(hit, best, torch.full_like(best, max_range)), torch.where(
        hit, idx, torch.full_like(idx, -1)
    )


def obb_surface_distance(
    qx: torch.Tensor, qy: torch.Tensor, obb: torch.Tensor
) -> torch.Tensor:
    """점 → OBB 표면 거리 (N,K). 내부면 0."""
    dx = qx.unsqueeze(1) - obb[..., 0]
    dy = qy.unsqueeze(1) - obb[..., 1]
    c, s = torch.cos(obb[..., 4]), torch.sin(obb[..., 4])
    lx = dx * c + dy * s
    ly = -dx * s + dy * c
    ex = (lx.abs() - obb[..., 2]).clamp(min=0)
    ey = (ly.abs() - obb[..., 3]).clamp(min=0)
    return torch.hypot(ex, ey)


def _wall_boxes(p: SceneParams) -> tuple[torch.Tensor, torch.Tensor]:
    """수조 벽 4면 (얇은 슬랩). 수면 위 1.0 m 까지 올라와 LiDAR 에 보인다."""
    t = C.TankCfg()
    L, W = t.length / 2, t.width / 2
    th = 0.3
    dev = p.berth_x.device
    n = p.berth_x.shape[0]
    boxes = torch.tensor(
        [
            [-L - th, -L, -W - th, W + th],  # x-
            [L, L + th, -W - th, W + th],  # x+
            [-L - th, L + th, -W - th, -W],  # y-
            [-L - th, L + th, W, W + th],  # y+ ← 버스가 있는 벽
        ],
        device=dev,
    ).view(1, 4, 4).expand(n, 4, 4)
    z = torch.tensor([-9.6, 1.0], device=dev).view(1, 1, 2).expand(n, 4, 2)
    return boxes, z


# =============================================================================
# 레이 vs AABB (2D 슬랩 기법) — 완전 벡터화
# =============================================================================
def _ray_aabb_2d(
    origin: torch.Tensor,  # (N,2)
    dirs: torch.Tensor,  # (N,R,2) 단위벡터
    boxes: torch.Tensor,  # (N,B,4) [xmin,xmax,ymin,ymax]
    max_range: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """각 광선의 최근접 교차거리와 맞은 박스 인덱스.

    Returns:
        dist: (N,R)  교차 없으면 max_range
        hit_idx: (N,R) 맞은 박스 인덱스, 없으면 -1
    """
    N, R, _ = dirs.shape
    B = boxes.shape[1]

    o = origin.view(N, 1, 1, 2)  # (N,1,1,2)
    d = dirs.view(N, R, 1, 2)  # (N,R,1,2)
    lo = torch.stack([boxes[..., 0], boxes[..., 2]], dim=-1).view(N, 1, B, 2)
    hi = torch.stack([boxes[..., 1], boxes[..., 3]], dim=-1).view(N, 1, B, 2)

    inv = 1.0 / torch.where(d.abs() < _EPS, torch.full_like(d, _EPS), d)
    t1 = (lo - o) * inv
    t2 = (hi - o) * inv
    t_near = torch.minimum(t1, t2).max(dim=-1).values  # (N,R,B)
    t_far = torch.maximum(t1, t2).min(dim=-1).values

    valid = (t_far >= t_near) & (t_far > 0)
    t_enter = torch.where(t_near > 0, t_near, torch.zeros_like(t_near))
    dist = torch.where(valid, t_enter, torch.full_like(t_enter, float("inf")))
    dist = torch.where(dist <= max_range, dist, torch.full_like(dist, float("inf")))

    best, idx = dist.min(dim=-1)  # (N,R)
    hit = torch.isfinite(best)
    return torch.where(hit, best, torch.full_like(best, max_range)), torch.where(
        hit, idx, torch.full_like(idx, -1)
    )


# =============================================================================
# LiDAR — Livox Mid-360
# =============================================================================
def lidar_scan(
    pos_xy: torch.Tensor,  # (N,2) 선체 중심
    yaw: torch.Tensor,  # (N,)
    p: SceneParams,
    n_bins: int | None = None,
    mount: "C.SensorMountCfg | None" = None,
) -> torch.Tensor:
    """수평 방위 스캔. 정규화 거리 (N, n_bins). 1.0 = 무반사(no return).

    mount 를 주면 장착 제원을 바꿔 시험할 수 있다(마스트 높이 민감도 분석 등).
    ※ dataclass 의 클래스 속성을 나중에 바꿔도 인스턴스 기본값은 __init__ 에
      이미 고정돼 있어 반영되지 않는다. 반드시 이 인자로 넘길 것.

    **수면과 수중 물체는 대상에서 제외한다.** 근거는 blueboat_cfg 주석 참조:
    스침각 정반사 + 905nm 흡수로 실제 Mid-360 은 이들을 관측하지 못한다.
    대상은 수면 위로 돌출한 고체(핑거 상부, 벽 상부)뿐이다.

    또한 Mid-360 의 수직시야 -7°~+52° 를 적용한다. LiDAR 보다 낮은 물체는
    (높이차)/tan(7°) 안쪽에서 시야를 벗어나 사라진다.
    """
    s = mount if mount is not None else C.SensorMountCfg()
    n_bins = n_bins or s.lidar_obs_h_bins
    N = pos_xy.shape[0]
    dev = pos_xy.device
    max_r = s.lidar_obs_max_range

    # 선체 기준 방위 격자 (선체와 함께 회전)
    az = torch.arange(n_bins, device=dev, dtype=torch.float32) * (2 * math.pi / n_bins)
    ang = yaw.view(N, 1) + az.view(1, n_bins)
    dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)  # (N,R,2)

    f_xy, f_z = _finger_boxes(p)
    w_xy, w_z = _wall_boxes(p)
    boxes = torch.cat([f_xy, w_xy], dim=1)  # (N,6,4)
    zs = torch.cat([f_z, w_z], dim=1)  # (N,6,2)
    # ※ 폐어구(_gear_obb)는 **의도적으로 제외**. 수면 아래라 LiDAR 에 원리적으로 안 보인다.

    dist, idx = _ray_aabb_2d(pos_xy, dirs, boxes, max_r)

    # --- 수직 시야 판정 ---
    h = s.lidar_pos[2]  # 수면 위 LiDAR 높이
    safe = idx.clamp(min=0)
    z_bot = torch.gather(zs[..., 0], 1, safe)  # (N,R)
    z_top = torch.gather(zs[..., 1], 1, safe)
    # 수면 위 부분만 남긴다 (수면 아래는 어차피 안 보임)
    z_bot = torch.clamp(z_bot, min=0.0)

    d_safe = dist.clamp(min=1e-3)
    ang_bot = torch.atan2(z_bot - h, d_safe)
    ang_top = torch.atan2(z_top - h, d_safe)
    fov_lo = math.radians(s.lidar_v_fov_min_deg)
    fov_hi = math.radians(s.lidar_v_fov_max_deg)
    in_fov = (ang_top >= fov_lo) & (ang_bot <= fov_hi) & (z_top > 0.0)

    valid = (idx >= 0) & in_fov & (dist >= s.lidar_min_range)
    out = torch.where(valid, dist / max_r, torch.ones_like(dist))
    return out.clamp(0.0, 1.0)


# =============================================================================
# FLS — 전방주시 소나 (기하 프록시)
# =============================================================================
def fls_scan(
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    p: SceneParams,
    n_beams: int | None = None,
) -> torch.Tensor:
    """전방 부채꼴 스캔. 정규화 거리 (N, n_beams). 1.0 = 무반사.

    **기하 프록시임을 명시한다.** 음향 물리(Lambert 산란, 빔패턴, Rayleigh 스펙클,
    다중경로)는 모델링하지 않는다. 본 연구 범위에서는 "수중 구조물의 거리-방위를
    준다"는 기능만 필요하고, 물리 기반 소나 모델은 별도 자산으로 존재한다.
    한계를 숨기지 않고 명시하는 편이 방어에 유리하다.

    대상: 수중 물체만 — 기초부, 핑거의 수면 아래 부분, 벽의 수면 아래 부분.
    """
    s = C.SensorMountCfg()
    n_beams = n_beams or s.fls_n_beams
    N = pos_xy.shape[0]
    dev = pos_xy.device
    max_r = s.fls_max_range

    zf = s.fls_pos[2]  # FLS 장착 깊이(음수)
    half = math.radians(s.fls_h_fov_deg / 2)
    rel = torch.linspace(-half, half, n_beams, device=dev)
    ang = yaw.view(N, 1) + rel.view(1, n_beams)
    dirs = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    # 벽·핑거는 축 정렬(AABB), 폐어구는 방향 있는 상자(OBB) 다. 따로 캐스트하고
    # 더 가까운 쪽을 채택한다. 하나의 코드로 합치면 OBB 회전이 AABB 에도 걸려 낭비다.
    f_xy, f_z = _finger_boxes(p)
    w_xy, w_z = _wall_boxes(p)
    aabb = torch.cat([f_xy, w_xy], dim=1)
    az_ = torch.cat([f_z, w_z], dim=1)
    d_a, i_a = _ray_aabb_2d(pos_xy, dirs, aabb, max_r)

    g_obb, g_z = _gear_obb(p)
    d_g, i_g = _ray_obb_2d(pos_xy, dirs, g_obb, max_r)

    use_gear = d_g < d_a
    dist = torch.where(use_gear, d_g, d_a)
    idx = torch.where(use_gear, i_g, i_a)

    # 수직 판정용 z 범위를 히트한 물체에서 가져온다
    safe_a, safe_g = i_a.clamp(min=0), i_g.clamp(min=0)
    zb = torch.where(use_gear, torch.gather(g_z[..., 0], 1, safe_g),
                     torch.gather(az_[..., 0], 1, safe_a))
    zt = torch.where(use_gear, torch.gather(g_z[..., 1], 1, safe_g),
                     torch.gather(az_[..., 1], 1, safe_a))
    z_bot, z_top = zb, torch.clamp(zt, max=0.0)  # 수면 위는 음향으로 안 보임

    d_safe = dist.clamp(min=1e-3)
    ang_bot = torch.atan2(z_bot - zf, d_safe)
    ang_top = torch.atan2(z_top - zf, d_safe)
    hf = math.radians(s.fls_v_fov_deg / 2)
    in_beam = (ang_top >= -hf) & (ang_bot <= hf) & (z_bot < 0.0)

    valid = (idx >= 0) & in_beam
    out = torch.where(valid, dist / max_r, torch.ones_like(dist))
    return out.clamp(0.0, 1.0)


# =============================================================================
# 도킹 목표와 충돌 판정
# =============================================================================
def docking_target(p: SceneParams) -> torch.Tensor:
    """(N,3) [x, y, yaw] 목표 자세. **항상 명목값이다.**

    이전 시나리오(벽 부착 기초부)에서는 기초부 돌출에 따라 목표를 뒤로 물렸다.
    폐어구는 벽이 아니라 **경로 위**에 있으므로 종점을 바꾸지 않는다.
    대신 가는 길을 바꿔야 한다 — 회피라는 행동이 생기고, 그것이 FLS 의 값어치다.
    덕분에 목표 계산이 단순해지고 성공 판정도 모든 에피소드에서 동일해진다.
    """
    y = torch.full_like(p.berth_x, WALL_Y - 0.50 - C.LOA / 2)
    yaw = torch.full_like(y, math.pi / 2)
    return torch.stack([p.berth_x, y, yaw], dim=-1)


def _boat_support_radius(yaw: torch.Tensor, dirx: torch.Tensor, diry: torch.Tensor) -> torch.Tensor:
    """선체 OBB 의 방향별 유효 반경 (지지함수).

    반경 0.55 원 근사는 1.2×0.93 m 선박을 지나치게 뚱뚱하게 만든다.
    실제로는 정렬 상태에서 횡방향 반폭이 0.465 m 이고, 비스듬하면 더 커진다.
    방향 (dirx,diry) 로의 지지반경 = (L/2)|d·u| + (B/2)|d·v| 로 정확히 계산한다.
    """
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    # u = 선수축, v = 좌현축
    du = (dirx * c + diry * s_).abs()
    dv = (-dirx * s_ + diry * c).abs()
    return (C.LOA / 2) * du + (C.BEAM / 2) * dv


def _obb_aabb_overlap(
    cx: torch.Tensor, cy: torch.Tensor, yaw: torch.Tensor, box: torch.Tensor
) -> torch.Tensor:
    """선체 OBB 와 축정렬 상자의 교차 여부 (분리축 정리, SAT). (N,) vs (N,K) → (N,K).

    box: (N,K,4) [xmin,xmax,ymin,ymax]
    원 근사와 달리 **정확하다** — 선박이 비스듬할 때 실제로 더 넓게 쓸리는 것을 반영한다.
    """
    bx = 0.5 * (box[..., 0] + box[..., 1])
    by = 0.5 * (box[..., 2] + box[..., 3])
    bhx = 0.5 * (box[..., 1] - box[..., 0])
    bhy = 0.5 * (box[..., 3] - box[..., 2])
    dx, dy = bx - cx.unsqueeze(1), by - cy.unsqueeze(1)

    c, s_ = torch.cos(yaw).unsqueeze(1), torch.sin(yaw).unsqueeze(1)
    hL, hB = C.LOA / 2, C.BEAM / 2
    sep = torch.zeros_like(dx, dtype=torch.bool)
    # 축 4개: 선체 u,v 와 세계 x,y
    for ax, ay, r_self in ((c, s_, hL), (-s_, c, hB)):
        proj = (dx * ax + dy * ay).abs()
        r_box = bhx * ax.abs() + bhy * ay.abs()
        sep |= proj > (r_self + r_box)
    for ax, ay, r_box_h in ((1.0, 0.0, bhx), (0.0, 1.0, bhy)):
        proj = (dx * ax + dy * ay).abs()
        r_self = hL * (c * ax + s_ * ay).abs() + hB * (-s_ * ax + c * ay).abs()
        sep |= proj > (r_self + r_box_h)
    return ~sep


def check_collision(pos_xy: torch.Tensor, yaw: torch.Tensor, p: SceneParams) -> dict:
    """충돌·감김 판정.

    폐어구는 **감김(entanglement)** 이다. 선체가 부서지는 것이 아니라 프로펠러에
    그물이 감겨 추진력을 잃는다. 소형 선박 무력화의 대표 원인이며,
    본 시뮬레이션에서는 즉시 실패로 처리한다.

    판정 기준면은 프로펠러 최심점(0.272 m)이다. 폐어구 상단은 항상 그보다 얕게
    설정되므로(0.05~0.20 m), **수평으로 겹치면 곧 감김**이다.
    """
    b = C.BerthCfg()
    t = C.TankCfg()

    # 프로펠러 위치(선체 후방)
    aft = b.prop_aft_offset
    px = pos_xy[:, 0] - aft * torch.cos(yaw)
    py = pos_xy[:, 1] - aft * torch.sin(yaw)

    # --- 폐어구: 판정 경계는 **그물 + 최소 안전 이격거리** ---
    #   실체에 닿아야 실패로 치면 정책이 물리적 한계까지 붙어 다니는 법을 배운다.
    #   안전거리를 규정하고 그 경계를 침범하면 실패로 본다.
    g_obb, _ = _gear_obb(p)  # (N,K,5)
    # 선체 표면 기준 이격 = (중심→그물 표면 거리) - (그물 방향으로의 선체 지지반경)
    d_hull = obb_surface_distance(pos_xy[:, 0], pos_xy[:, 1], g_obb)  # (N,K)
    vx = g_obb[..., 0] - pos_xy[:, :1]
    vy = g_obb[..., 1] - pos_xy[:, 1:2]
    vn = torch.hypot(vx, vy).clamp(min=1e-6)
    r_boat = _boat_support_radius(yaw.unsqueeze(1), vx / vn, vy / vn)
    gap_hull = d_hull - r_boat
    gap_prop = obb_surface_distance(px, py, g_obb) - 0.25
    clearance = torch.minimum(gap_hull.min(dim=1).values, gap_prop.min(dim=1).values)
    entangled = clearance < b.gear_safety_margin

    # --- 핑거 접촉: 선체 OBB vs 핑거 상자, 분리축 정리로 정확히 판정 ---
    f_xy, _ = _finger_boxes(p)
    finger_hit = _obb_aabb_overlap(pos_xy[:, 0], pos_xy[:, 1], yaw, f_xy).any(dim=1)

    # --- 벽: 선수 끝 기준 (원 근사는 성공 영역을 잠식한다) ---
    bow_y = pos_xy[:, 1] + (C.LOA / 2) * torch.sin(yaw)
    wall_hit = (
        (bow_y > WALL_Y)
        | (pos_xy[:, 1] < -t.width / 2 + 0.6)
        | (pos_xy[:, 0].abs() > t.length / 2 - 0.6)
    )

    return {
        "entangled": entangled,
        "finger_hit": finger_hit,
        "wall_hit": wall_hit,
        "any": entangled | finger_hit | wall_hit,
        # 이격거리: 그물 표면에서 선체(또는 프로펠러) 표면까지. 음수면 실체 접촉.
        # 학습이 진행되면 이 값이 위에서 gear_safety_margin 으로 수렴해야 한다.
        "clearance": clearance,
    }
