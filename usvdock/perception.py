"""LiDAR 기반 버스(정박지) 검출 — 세 Arm 이 **공유**하는 인지 프론트엔드.

왜 공유하는가:
  초기에는 RL 이 원시 스캔 72빈에서 직접 버스를 찾아내게 두었으나, 도킹 보너스가
  희소해서(횡 0.25 m·선수각 12°·10스텝 유지) 학습 신호가 잡히지 않았다.
  120 iter 동안 보상이 -105 → -131 로 악화되고 에피소드는 대부분 시간초과였다.

  해법은 Arm A(고전 파이프라인)가 쓰는 검출기를 **세 Arm 모두에게** 주는 것이다.
  정답 위치가 아니라 스캔에서 추정한 값이므로 공정성이 유지되고,
  비교 구도가 오히려 선명해진다 — **같은 인지 프론트엔드, 다른 백엔드**:

      Arm A  검출 → PID
      Arm B  검출 + 원시 LiDAR → RL
      Arm C  검출 + 원시 LiDAR + FLS → RL

  즉 A vs B 는 순수하게 제어기 차이, B vs C 는 순수하게 수중 정보 차이가 된다.

★ 무엇을 하고, 무엇을 하지 않는가 (발표 질의응답 대비)

  하는 것 : **단일 프레임 기하 추출.** 한 번의 스캔에서 벽면과 핑거를 찾아
            버스 중심의 상대 위치를 낸다.
  안 하는 것: SLAM, 점유격자 매핑, 시간적 누적/추적(칼만·파티클), 자기위치추정.
            정책도 MLP 라 기억이 없다 — 매 스텝 현재 스캔만 본다.

  이렇게 한 이유: 본 과제는 **종단 접근**(6~12 m)이고 버스가 직접 관측되므로
  전역 지도를 만들 이유가 없다. 다만 한계는 분명하다 — 대상이 시야에서
  사라지면 즉시 잊는다. 이는 향후 과제로 명시한다(순환 정책 또는 국소 점유격자).

★ 좌표 처리 — 참 위치를 쓰지 않는다

  초기 구현은 벽까지 거리를 세계좌표 상수에서 가져와 정책에 넘겼다.
  실기에서 측정 없이 얻을 수 없는 값이라 정보 누수였다.
  현재는 **벽 거리도 스캔에서 추정**한다. 필요한 외부 정보는 선수각(yaw)뿐이며,
  이는 IMU/지자기 센서로 실제 관측 가능한 양이다.

검출 원리:
  핑거는 벽에서 안쪽으로 돌출한 유일한 구조물이다. 스캔점을 직교좌표로 펴서
  "벽보다 안쪽이고 핑거 길이 안"인 점만 남기면 그것이 핑거 반사다.
  좌우 극단 x 의 중점이 버스 중심이 된다.
  ※ 정답 좌표(berth_x)를 쓰지 않는다. 쓰면 인지 문제가 사라진다.
"""

from __future__ import annotations

import math

import torch

from . import blueboat_cfg as C
from . import geometry as G

_NAN = float("nan")


def detect_berth(
    pos_xy: torch.Tensor,
    yaw: torch.Tensor,
    scene: "G.SceneParams",
    mount: C.SensorMountCfg | None = None,
    scan: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """스캔에서 버스를 검출한다. **참 위치를 쓰지 않는다.**

    선수각(yaw)만 외부 입력으로 쓴다 — IMU/지자기로 실제 관측 가능한 양이다.
    반환값은 전부 "세계축 정렬, 선체 중심" 상대 좌표계의 값이다.

    Returns:
        rel_x: (N,) 버스 중심의 상대 x (벽을 따라). 실패 시 NaN
        wall_d: (N,) 벽까지의 상대 거리 (세계 y 방향). 실패 시 NaN
        ok:    (N,) 검출 성공 여부
    """
    s = mount if mount is not None else C.SensorMountCfg()
    # scan 을 받으면 재계산하지 않는다. 한 스텝에서 관측·검출·PID 가 같은 스캔을
    # 각자 다시 뜨면 낭비가 크다(평가에서 스텝당 LiDAR 스캔이 3회 돌고 있었다).
    R = s.lidar_det_h_bins
    if scan is None:
        scan = G.lidar_scan(pos_xy, yaw, scene, n_bins=R, mount=s)
    N = pos_xy.shape[0]
    rng = scan * s.lidar_obs_max_range

    az = torch.arange(R, device=scan.device, dtype=torch.float32) * (2 * math.pi / R)
    ang = yaw.view(N, 1) + az.view(1, R)
    # 세계축 정렬, 선체 중심 상대 좌표 (참 위치가 들어가지 않는다)
    rx = rng * torch.cos(ang)
    ry = rng * torch.sin(ang)

    hit = scan < 0.999
    # 센서 기반 관심영역: 전방(세계 +y)이고 측방 ROI_X 이내.
    # 측벽을 배제하기 위한 것이며, 수조 도면이 아니라 센서 범위로 정의한다.
    #   위치 기반 ROI 는 실패했다: |rx|<8 → 78.8%, |rx|<14 → 31.1%.
    #   넓힐수록 나빠진 이유는 **측벽 모서리의 ry 가 버스 벽보다 클 수 있기** 때문이다
    #   (측벽은 y=-10.15~10.15 로 버스 벽 10.0 보다 약간 더 뻗어 있다).
    #   그러면 "가장 먼 전방 반사 = 벽" 추정이 측벽을 집는다.
    #
    #   → **광선 방향**으로 제한한다. 세계 +y(벽 법선) 기준 ±45° 부채꼴만 본다.
    #     이 방향의 광선은 버스 벽 아니면 핑거에 맞는다. 위치가 아니라 방향으로
    #     자르는 것이 이 문제의 올바른 형태다("벽 쪽을 본다").
    #   부채꼴 폭은 원거리 실측으로 정했다: ±30° 91.9%, **±40° 97.4%**, ±45° 95.0%.
    #
    # ★ 근거리 예외 — 고정 부채꼴만 쓰면 **종단에서 검출이 끊긴다.**
    #   배가 벽 가까이(y=8.5) 옆으로(x=1.85) 붙으면 핑거가 세계 +y 기준
    #   atan2(-0.875, 1.0) = -41° 로 ±40° 밖으로 나간다.
    #   실측 검출 지도(벽까지 거리 × 횡오프셋):
    #       벽 8.0 m: 전부 검출        벽 3.0 m: x=1 부터 실패
    #       벽 1.5 m: x=1 부터 실패    벽 1.0 m: **정중앙에서도 실패**
    #   정밀도가 가장 필요한 구간이 통째로 사각이었다. 실제로 정책이 목표 1.9 m 앞에서
    #   검출을 잃고 40초를 배회했다(검출 성공률 23.4%).
    #
    #   → 가까운 반사(5 m 이내)는 부채꼴과 무관하게 받는다. 근거리에서 잡히는 것은
    #     측벽이 아니라 버스 구조물이므로 오염 위험이 없다.
    sector = (torch.remainder(ang - math.pi / 2 + math.pi, 2 * math.pi) - math.pi).abs()
    near = rng < 5.0
    fwd = hit & (ry > 0.3) & ((sector < math.radians(40.0)) | near)

    NEG = torch.full_like(ry, -1e9)
    # 버스 벽 = 전방에서 가장 먼 반사면 (그 뒤에는 아무것도 없다)
    wall_d = torch.where(fwd, ry, NEG).max(dim=1).values
    wall_ok = wall_d > 0.5

    b = C.BerthCfg()
    # 1단계: 핑거 대역 = 벽에서 안쪽으로 돌출한 반사
    band = (
        fwd
        & (ry > wall_d.view(N, 1) - b.pile_length - 0.3)
        & (ry < wall_d.view(N, 1) - 0.25)
    )
    # 2단계: 대역 안 rx 중앙값 주변 2 m 만 남긴다.
    #   부채꼴 안에서도 먼 측벽 반사가 이 대역에 섞여 들어와 span 을 부풀렸다.
    #   (실패의 100% 가 span 조건이었다: ±45° 에서 40% 탈락)
    #   중앙값은 소수 이상치에 강건하므로 한 번의 정제로 충분하다.
    cxn = torch.where(band, rx, torch.full_like(rx, _NAN))
    med = torch.nanmedian(cxn, dim=1).values
    fing = band & ((rx - med.view(N, 1)).abs() < 2.0)
    n = fing.sum(dim=1)

    POS = torch.full_like(rx, 1e9)
    x_min = torch.where(fing, rx, POS).min(dim=1).values
    x_max = torch.where(fing, rx, NEG).max(dim=1).values
    center = 0.5 * (x_min + x_max)
    span = x_max - x_min

    ok = wall_ok & (n >= 2) & (span > 0.8) & (span < 3.5)
    nan = torch.full_like(center, _NAN)
    return torch.where(ok, center, nan), torch.where(ok, wall_d, nan), ok


def berth_features(
    pos_xy: torch.Tensor, yaw: torch.Tensor, scene: "G.SceneParams",
    scan: torch.Tensor | None = None,
) -> torch.Tensor:
    """정책 입력용 압축 특징 (N,5).

    [선체 전방 상대거리, 좌현 상대거리, 선수각오차 cos, sin, 검출 유효 플래그]
    모두 스캔에서 나온 값이다. 검출 실패 시 0 을 넣고 플래그로 알린다 —
    정책이 "지금 안 보인다"를 구분할 수 있어야 한다.
    """
    rel_x, wall_d, ok = detect_berth(pos_xy, yaw, scene, scan=scan)
    N = pos_xy.shape[0]

    # 명목 정박점: 벽에서 0.5 m + 선체 반길이 앞.
    # 수중 사정은 LiDAR 로 알 수 없다 — 그것이 미션2 의 핵심이다.
    dx = rel_x
    dy = wall_d - 0.50 - C.LOA / 2

    c, s = torch.cos(yaw), torch.sin(yaw)
    bx = dx * c + dy * s  # 선체 전방
    by = -dx * s + dy * c  # 선체 좌현
    yaw_err = (math.pi / 2 - yaw + math.pi) % (2 * math.pi) - math.pi

    z = torch.zeros(N, device=pos_xy.device)
    return torch.stack(
        [
            torch.where(ok, bx / 20.0, z),
            torch.where(ok, by / 20.0, z),
            torch.where(ok, torch.cos(yaw_err), z),
            torch.where(ok, torch.sin(yaw_err), z),
            ok.float(),
        ],
        dim=-1,
    )
