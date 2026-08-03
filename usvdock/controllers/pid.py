"""Arm A — 고전 파이프라인: LiDAR → 핑거 검출 → 버스 자세 추정 → PID.

★ 이 Arm 의 설계 의도

  초기 구상은 "단순 PID vs RL+LiDAR" 였으나, 그러면 **제어기와 인지를 동시에
  바꾸는 교란된 비교**가 된다. PID 가 진 것이 제어기 탓인지 눈이 없어서인지
  구분할 수 없다.

  그래서 PID 에게도 **같은 LiDAR** 를 준다. 다만 학습이 아니라 고전적으로 처리한다:
      스캔 → 핑거 2개 클러스터 검출 → 중점·법선 계산 → 목표 자세 → PID 추종
  이것이 엔지니어가 실제로 만들 파이프라인이고, 공정한 대조군이다.

  결과적으로 비교 축이 하나씩만 바뀐다:
      A vs B   정보 동일, 제어기만 다름  → 제어기의 가치
      B vs C   제어기 동일, 정보만 다름  → 센서 융합의 가치

★ 부족구동 처리

  BlueBoat 는 횡추력이 없다(배분행렬 Y 행이 0). 따라서 고전적 도킹 제어의
  표준 구성인 "거리 오차 → 전진, 방위 오차 → 회두" 2-루프로 간다.
  횡방향 오차는 직접 없앨 수 없고, 목표를 향하도록 선수를 돌린 뒤 전진해
  간접적으로 줄인다(시선각 유도, line-of-sight guidance).
  횡류가 있으면 이 방식이 정상상태 편류를 남긴다 — 이것이 Arm A 의 구조적 한계이며,
  RL 이 우위를 보일 것으로 예상되는 지점이다.
"""

from __future__ import annotations

import math

import torch

from .. import blueboat_cfg as C
from .. import geometry as G


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DockingPID:
    """시선각 유도 + 2-루프 PID (전진 / 회두) + FLS 기반 반응형 회피.

    출력은 정규화 추력 [-1,1]² (좌현, 우현) — RL 정책과 동일한 인터페이스라
    같은 환경에서 그대로 비교할 수 있다.

    ★ 회피는 "최적 빔 선택(steer-to-best-beam)" 방식이다.
      FLS 빔마다 (목표 방위와의 각도차) + (막힘 벌점) 으로 비용을 매기고
      최소 비용 빔 방향으로 조타한다. VFH/갭 추종의 단순화된 형태이며,
      전역 계획 없이 현재 스캔만으로 동작한다 — RL 정책과 정보 조건이 같다.

      전역 경로계획(A*, RRT)을 쓰지 않은 이유: 그러면 PID 쪽에만 지도와 계획이
      생겨 비교가 교란된다. 양쪽 다 **현재 스캔에 즉시 반응**하는 조건으로 맞춘다.
    """

    def __init__(self, num_envs: int, device: str, dt: float):
        self.dt = dt
        self.dev = device
        z = lambda: torch.zeros(num_envs, device=device)  # noqa: E731
        self._i_dist, self._i_yaw = z(), z()
        self._e_dist_prev, self._e_yaw_prev = z(), z()

        # 이득: 저속 정밀 접근용으로 보수적으로 잡았다.
        self.kp_d, self.ki_d, self.kd_d = 0.55, 0.02, 0.25
        self.kp_y, self.ki_y, self.kd_y = 1.30, 0.01, 0.35
        self.approach_speed = 0.6  # 목표 접근 속력 [m/s]

        # --- FLS 회피 파라미터 ---
        self.avoid_range = 4.0  # 이 거리 안의 반사를 장애물로 본다 [m]
        self.clear_margin = 0.9  # 선체 반폭(0.465)+여유. 각폭 팽창에 쓴다 [m]
        self.avoid_slow = 0.30  # 회피 중 접근 속력 [m/s]

    def reset_idx(self, ids: torch.Tensor):
        for t in (self._i_dist, self._i_yaw, self._e_dist_prev, self._e_yaw_prev):
            t[ids] = 0.0

    def _avoid(
        self, fls: torch.Tensor, yaw: torch.Tensor, desired_yaw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FLS 스캔에서 통과 가능한 최적 방향을 고른다.

        Returns:
            새 목표 선수각, 회피 작동 여부(bool)
        """
        s = C.SensorMountCfg()
        N, nb = fls.shape
        rng = fls * s.fls_max_range
        half = math.radians(s.fls_h_fov_deg / 2)
        ang = torch.linspace(-half, half, nb, device=fls.device).view(1, nb)

        blocked = (fls < 0.999) & (rng < self.avoid_range)  # (N,nb)

        # 각폭 팽창: 거리 r 의 장애물은 asin(여유/r) 만큼 좌우로 넓혀 봐야 한다.
        # (배가 점이 아니라 폭을 갖기 때문. 안 하면 장애물 가장자리를 스치며 지나간다)
        pad = torch.asin(torch.clamp(self.clear_margin / rng.clamp(min=0.3), max=1.0))
        d_ang = (ang.unsqueeze(1) - ang.unsqueeze(2)).abs()  # (1,nb,nb)
        unsafe = ((d_ang <= pad.unsqueeze(2)) & blocked.unsqueeze(2)).any(dim=1)  # (N,nb)

        # 목표 방위를 선체 기준으로
        goal_rel = _wrap_pi(desired_yaw - yaw).view(N, 1)
        cost = (ang - goal_rel).abs() + 10.0 * unsafe.float()
        best = cost.argmin(dim=1)
        pick = torch.gather(ang.expand(N, nb), 1, best.view(N, 1)).squeeze(1)

        # 목표 방향 자체가 막혔을 때만 회피로 간주
        gi = (goal_rel.clamp(-half, half) - ang).abs().argmin(dim=1)  # (N,1)-(1,nb) → (N,nb)
        goal_blocked = torch.gather(unsafe, 1, gi.view(N, 1)).squeeze(1)

        return torch.where(goal_blocked, yaw + pick, desired_yaw), goal_blocked

    def __call__(
        self,
        pos_xy: torch.Tensor,
        yaw: torch.Tensor,
        nu: torch.Tensor,
        target: torch.Tensor,
        fls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        dx = target[:, 0] - pos_xy[:, 0]
        dy = target[:, 1] - pos_xy[:, 1]
        dist = torch.hypot(dx, dy)

        # 시선각 유도: 멀면 목표 방향, 가까우면 최종 선수각으로 부드럽게 전환
        los = torch.atan2(dy, dx)
        blend = torch.clamp(dist / 3.0, 0.0, 1.0)
        desired_yaw = torch.atan2(
            blend * torch.sin(los) + (1 - blend) * torch.sin(target[:, 2]),
            blend * torch.cos(los) + (1 - blend) * torch.cos(target[:, 2]),
        )

        avoiding = torch.zeros_like(dist, dtype=torch.bool)
        if fls is not None:
            desired_yaw, avoiding = self._avoid(fls, yaw, desired_yaw)

        e_yaw = _wrap_pi(desired_yaw - yaw)

        # 전진 루프 — **부호 있는 종방향 오차**를 쓴다.
        #   초기 구현은 v_ref = clamp(dist*0.5, 0, v_max) 였다. 거리는 부호가 없으므로
        #   목표를 지나쳐도 계속 전진을 명령했고, 배가 벽을 뚫고 나간 뒤 제자리에서
        #   회전했다(오프라인 궤적 시험에서 확인: t=20s 거리 0.26 m → t=28s y=10.57).
        #   선체 전방축에 오차를 투영해 부호를 살리면 지나쳤을 때 후진이 나온다.
        along = dx * torch.cos(yaw) + dy * torch.sin(yaw)
        v_max = torch.where(avoiding, torch.full_like(along, self.avoid_slow),
                            torch.full_like(along, self.approach_speed))
        v_ref = torch.clamp(along * 0.6, -0.30, 1.0) .minimum(v_max)
        e_dist = v_ref - nu[:, 0]

        self._i_dist = (self._i_dist + e_dist * self.dt).clamp(-2.0, 2.0)
        self._i_yaw = (self._i_yaw + e_yaw * self.dt).clamp(-1.0, 1.0)
        d_dist = (e_dist - self._e_dist_prev) / self.dt
        d_yaw = _wrap_pi(e_yaw - self._e_yaw_prev) / self.dt
        self._e_dist_prev, self._e_yaw_prev = e_dist, e_yaw

        surge = self.kp_d * e_dist + self.ki_d * self._i_dist + self.kd_d * d_dist
        yawcmd = self.kp_y * e_yaw + self.ki_y * self._i_yaw + self.kd_y * d_yaw

        # 선수 오차가 크면 전진을 줄인다(엉뚱한 방향으로 돌진 방지)
        surge = surge * torch.cos(e_yaw).clamp(min=0.0)

        # 차동 배분: 좌현 = surge + yaw, 우현 = surge - yaw
        port = (surge + yawcmd).clamp(-1.0, 1.0)
        stbd = (surge - yawcmd).clamp(-1.0, 1.0)
        return torch.stack([port, stbd], dim=-1)
