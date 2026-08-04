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
    """제대로 된 도킹 파이프라인 — 경유점 2단계 진입 + 제동 프로파일 + 횡편차 루프 + 회피 이력.

    출력은 정규화 추력 [-1,1]² (좌현, 우현) — RL 정책과 동일한 인터페이스라
    같은 환경에서 그대로 비교할 수 있다.

    ★ 왜 다시 썼는가 — 이전 구현의 실측 결함 2건

      결함 1. 회피가 **접안 구조물을 장애물로 취급**했다. FLS 는 폐어구뿐 아니라
        핑거·벽도 돌려준다. 목표는 슬립 안쪽(y=8.9), 구조물은 y=7.5~10.0 이므로
        부두 4 m 안에서 목표 방위가 항상 막힌 것으로 판정되어 반대편으로 조타했다.
        폐어구가 **하나도 없는** 장면에서도 횡오차 0.02 m 로 정렬해 놓고
        2.31 m 를 남긴 채 선수 180°, 속력 0.000 으로 교착했다.
        → 검출된 선석 기하로 알려진 구조물 빔을 제외(_avoid 참조).

      결함 2. **진입 회랑이 없었다.** 목표점 하나만 보고 시선각 유도로 들어가니
        슬립에 대각선으로 파고들어 핑거 끝을 스쳤다. 실측 충돌 시점:
        선수 124°(필요 90°), 속력 0.51 m/s 순항 중, 목표까지 2.46 m.
        감속 구간(종방향 1 m 이내)에 **진입하기도 전에** 부딪힌 것이다.
        → 아래 4개 규칙으로 재구성.

    ★ 설계 (실제 도킹 제어기의 표준 구성)

      ① 2단계 경유점 진입
         슬립 중심선 위, 입구 바깥 `entry_standoff` 지점에 경유점 W 를 둔다.
         W 에 도달하고 선수가 정렬되기 전에는 슬립에 들어가지 않는다.
         부족구동선은 횡이동이 안 되므로 **정렬 후 직진**이 유일한 안전한 진입이다.

      ② 제동거리 기반 감속
         v_ref = min(v_max, sqrt(2·a_brake·along)).
         이전 구현은 v_ref = along×0.6 (상한 0.6) 이라 종방향 1 m 안에 들어와야
         감속이 시작됐다. 제동거리를 알고 미리 줄인다.

      ③ 최종 구간 횡편차 루프
         중심선 이탈을 선수각 오프셋(게걸음각)으로 잡는다. 횡추력이 없으므로
         이것이 유일한 수단이다. 유속이 있으면 정상상태 게걸음각이 남는다.

      ④ 회피 이력(hysteresis)
         우회 방향을 한 번 정하면 `avoid_hold_s` 동안 그 쪽을 유지한다.
         이전 구현은 매 스텝 최적 빔을 다시 골라, 틀면 목표가 다시 열리고
         되돌아오는 극한 주기에 빠졌다(실측 왕복 28회).

    ★ 전역 경로계획(A*, RRT)은 여전히 쓰지 않는다. 회피는 현재 스캔에 반응한다.
      다만 **선석은 이미 검출된 목표**이므로 그 기하를 쓰는 것은 계획이 아니라 유도다.
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
        self.approach_speed = 0.6  # 순항 접근 속력 [m/s]

        # --- ① 경유점 진입 ---
        # 대기선은 입구에서 이만큼 앞. 0.80 으로 뒀다가 핑거에 닿았다:
        #   제동거리 v²/2a = 0.45²/0.24 = 0.84 m + 게걸음 자세에서 선수부가
        #   y 로 0.5 m 더 뻗는다 → 최소 1.4 m 는 필요하다.
        self.entry_standoff = 1.70  # [m]
        self.los_lookahead = 1.8  # LOS 전방주시거리 Δ [m]. 작을수록 공격적으로 복귀
        self.entry_x_tol = 0.35  # 중심선 정렬 허용 [m]
        self.entry_yaw_tol = math.radians(20.0)  # 선수 정렬 허용
        self.entry_abort_x = 0.90  # 이만큼 벗어나면 다시 경유점 단계로

        # --- ② 제동 프로파일 ---
        # 목표 허용오차(종방향 0.35 m, 속력 0.20 m/s)를 만족하려면
        # along=0.2 m 에서 v≈0.2 여야 한다 → a = v²/(2·along) ≈ 0.10
        self.brake_accel = 0.12  # [m/s²]

        # --- ③ 횡편차 루프 ---
        # ILOS(적분 시선각): 정상 유속을 적분항으로 상쇄한다.
        #   비적분 LOS 는 유속을 **원리적으로** 못 이긴다. 게걸음각 δ 로 유속 V 를
        #   상쇄하려면 u·sin(δ)=V 이어야 하는데, δ=atan2(x_err, Δ) 는 x_err 가
        #   커져야만 커지므로 정상 편차가 남는다. 실측(유속 0.30): 횡편차가
        #   +0.03 → +1.62 m 로 단조 증가하며 되잡지 못했다.
        self.los_ki = 0.35  # 적분 이득 [1/s]
        self.los_i_max = 2.5  # 적분항 상한 [m] (와인드업 방지)
        self.max_crab = math.radians(60.0)  # 게걸음각 상한. asin(0.45/0.5)=64° 를 고려

        # --- ④ FLS 회피 ---
        self.avoid_range = 4.0  # 이 거리 안의 반사를 장애물로 본다 [m]
        self.clear_margin = 0.9  # 선체 반폭(0.465)+여유. 각폭 팽창에 쓴다 [m]
        self.avoid_slow = 0.35  # 회피 중 접근 속력 [m/s]
        # 조종성 확보 최소 속력(maintain way). 부족구동선은 **전진해야만 횡방향을
        # 바꿀 수 있다.** 종방향 루프가 "멈춰"라고 하면 횡편차를 잡을 수단도 사라진다.
        # 실측: 대기선을 0.18 m 지나치자 후진 지령이 걸렸는데, 복귀 침로(150°)에서
        # 후진은 +x 로 밀려나는 방향이라 x 오차가 +1.4 → +14.4 m 로 발산했다.
        self.v_min_transit = 0.30  # [m/s]
        self.avoid_hold_s = 1.5  # 우회 방향 유지 시간 [s]

        # 상태
        self._in_final = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._avoid_hold = z()  # 남은 유지 시간 [s]
        self._avoid_dir = z()  # 우회 방향 부호 (-1 우현쪽 / +1 좌현쪽)
        self._i_ct = z()  # 횡편차 적분항 [m·s]

    def reset_idx(self, ids: torch.Tensor):
        for t in (self._i_dist, self._i_yaw, self._e_dist_prev, self._e_yaw_prev,
                  self._avoid_hold, self._avoid_dir, self._i_ct):
            t[ids] = 0.0
        self._in_final[ids] = False

    def _avoid(
        self, fls: torch.Tensor, yaw: torch.Tensor, desired_yaw: torch.Tensor,
        pos_xy: torch.Tensor, target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FLS 스캔에서 통과 가능한 최적 방향을 고른다.

        ★ **알려진 접안 구조물은 회피 대상에서 뺀다.**

          초기 구현은 FLS 반사를 전부 장애물로 봤다. 그런데 FLS 는 폐어구뿐 아니라
          핑거와 벽도 돌려준다. 목표는 슬립 **안쪽**(y=8.9)이고 구조물은 y=7.5~10.0 이므로,
          배가 부두 4 m 안으로 들어오면 목표 방위가 **항상** goal_blocked 이 되고
          회피가 부두 반대편 빈 물로 조타했다. 거기에 surge *= cos(e_yaw) 가 겹쳐
          선수오차 90°에서 전진 추력이 0 이 되어 완전히 멈췄다.

          실측(폐어구가 **하나도 없는** 장면): 횡오차 0.02 m 로 정렬해 놓고
          접근축 2.31 m 를 남긴 채 선수 180°, 속력 0.000 으로 교착. 유속·장애물과
          무관한 결정론적 실패였다 — 즉 **목적지를 장애물로 인식하고 도망친 것**이다.

          이 상태의 PID 를 대조군으로 쓰면 "RL 이 이겼다"가 아니라
          "고장 난 베이스라인을 이겼다"가 된다.

          → 검출된 선석 기하로 **알려진 구조물에 해당하는 빔을 가려내고**,
            나머지(지도에 없는 반사)만 위험으로 본다. 실제 엔지니어가 만들 파이프라인이고,
            RL 에 없는 정보를 주는 것도 아니다(선석 위치는 LiDAR 검출로 이미 갖고 있다).

          한계: 슬립 **안쪽**에 폐어구가 떠 있으면 가려져 못 본다. 본 연구의 장면
          분포에서는 폐어구가 y∈[3.0, 6.5], 구조물이 y≥7.5 로 겹치지 않는다.

        Returns:
            새 목표 선수각, 회피 작동 여부(bool)
        """
        s = C.SensorMountCfg()
        N, nb = fls.shape
        rng = fls * s.fls_max_range
        half = math.radians(s.fls_h_fov_deg / 2)
        ang = torch.linspace(-half, half, nb, device=fls.device).view(1, nb)

        # --- 알려진 구조물 빔 가려내기 ---
        # 빔 끝점을 세계좌표로 옮긴 뒤, 검출된 선석의 footprint 안이면 구조물로 본다.
        b = C.BerthCfg()
        berth_x = target[:, 0].view(-1, 1)
        # 목표 y = WALL_Y - 0.50 - LOA/2 이므로 역산한다(참값을 쓰지 않는다)
        wall_y = (target[:, 1] + 0.50 + C.LOA / 2).view(-1, 1)
        th = (yaw.view(-1, 1) + ang)  # 세계 기준 빔 방위
        ex = pos_xy[:, 0].view(-1, 1) + rng * torch.cos(th)
        ey = pos_xy[:, 1].view(-1, 1) + rng * torch.sin(th)
        half_span = b.pile_gap / 2 + b.pile_diameter + 0.60  # 핑거 바깥까지 + 여유
        known = (
            (ey > wall_y - b.pile_length - 0.40) & ((ex - berth_x).abs() < half_span)
        ) | (ey > wall_y - 0.40)  # 벽면은 x 무관
        blocked = (fls < 0.999) & (rng < self.avoid_range) & (~known)  # (N,nb)

        # 각폭 팽창: 거리 r 의 장애물은 asin(여유/r) 만큼 좌우로 넓혀 봐야 한다.
        # (배가 점이 아니라 폭을 갖기 때문. 안 하면 장애물 가장자리를 스치며 지나간다)
        pad = torch.asin(torch.clamp(self.clear_margin / rng.clamp(min=0.3), max=1.0))
        d_ang = (ang.unsqueeze(1) - ang.unsqueeze(2)).abs()  # (1,nb,nb)
        unsafe = ((d_ang <= pad.unsqueeze(2)) & blocked.unsqueeze(2)).any(dim=1)  # (N,nb)

        # 목표 방위를 선체 기준으로
        goal_rel = _wrap_pi(desired_yaw - yaw).view(N, 1)

        # 목표 방향 자체가 막혔을 때만 회피로 간주
        gi = (goal_rel.clamp(-half, half) - ang).abs().argmin(dim=1)
        goal_blocked = torch.gather(unsafe, 1, gi.view(N, 1)).squeeze(1)

        # ④ 이력: 새로 막히면 우회 방향을 정하고 avoid_hold_s 동안 **유지**한다.
        #   이전 구현은 매 스텝 최적 빔을 다시 골랐다. 틀면 목표 방향이 다시 열리고
        #   곧바로 되돌아와 극한 주기에 빠진다(실측 왕복 28회, 30초 배회).
        fresh = goal_blocked & (self._avoid_hold <= 0.0)
        cost0 = (ang - goal_rel).abs() + 10.0 * unsafe.float()
        side = torch.sign(
            torch.gather(ang.expand(N, nb), 1, cost0.argmin(dim=1).view(N, 1)).squeeze(1)
        )
        self._avoid_dir = torch.where(fresh, torch.where(side == 0, torch.ones_like(side), side),
                                      self._avoid_dir)
        self._avoid_hold = torch.where(
            goal_blocked, torch.full_like(self._avoid_hold, self.avoid_hold_s),
            (self._avoid_hold - self.dt).clamp(min=0.0),
        )
        avoiding = self._avoid_hold > 0.0

        # 정해진 쪽 빔만 후보로 둔다(반대쪽으로 되틀지 않게)
        same_side = (ang * self._avoid_dir.view(N, 1)) >= -1e-6
        cost = (ang - goal_rel).abs() + 10.0 * unsafe.float() + 10.0 * (~same_side).float()
        pick = torch.gather(ang.expand(N, nb), 1, cost.argmin(dim=1).view(N, 1)).squeeze(1)

        return torch.where(avoiding, yaw + pick, desired_yaw), avoiding

    def __call__(
        self,
        pos_xy: torch.Tensor,
        yaw: torch.Tensor,
        nu: torch.Tensor,
        target: torch.Tensor,
        fls: torch.Tensor | None = None,
    ) -> torch.Tensor:
        b = C.BerthCfg()
        berth_x = target[:, 0]
        # 목표 y = WALL_Y - 0.50 - LOA/2 이므로 벽·입구를 역산한다(참값을 쓰지 않는다)
        wall_y = target[:, 1] + 0.50 + C.LOA / 2
        entrance_y = wall_y - b.pile_length  # 슬립 입구
        final_yaw = target[:, 2]

        # ── ① 중심선 경로추종 + 정렬 게이트 ──────────────────────────────
        # 처음엔 "슬립 입구 앞 경유점 W 로 간다"로 짰는데 **지나치면 되돌아갔다.**
        # 조준점이 뒤에 놓이니 선수가 뒤로 돌고(실측 141°→171°→-160°) 정렬 조건이
        # 영원히 성립하지 않아 입구 앞에서 40초간 요동쳤다.
        # → 이산 경유점 대신 **중심선 경로추종**(Fossen LOS)으로 바꾼다.
        #   지나침이라는 개념 자체가 없어진다.
        x_err = pos_xy[:, 0] - berth_x  # 중심선 횡편차 (우현 +)

        # ③ LOS 유도: 전방주시거리 Δ 앞의 중심선 위 점을 향한다.
        #   선수 90°+δ 는 (-sinδ, cosδ) 방향이므로 우현 이탈(x_err>0)에 δ>0 이어야 한다.
        #   부족구동선은 횡추력이 없어 이 선수 오프셋(게걸음각)이 유일한 횡방향 수단이다.
        # ILOS: 적분항이 정상 유속을 상쇄한다. 순수 LOS 는 유속에 정상 편차를 남긴다.
        self._i_ct = (self._i_ct + x_err * self.dt).clamp(-self.los_i_max, self.los_i_max)
        crab = torch.atan2(x_err + self.los_ki * self._i_ct,
                           torch.full_like(x_err, self.los_lookahead))
        desired_yaw = final_yaw + crab.clamp(-self.max_crab, self.max_crab)

        # 정렬 게이트: 중심선과 선수가 모두 맞기 전에는 슬립에 들어가지 않는다.
        #   부족구동선은 슬립 안에서 횡방향 수정이 불가능하므로,
        #   비스듬히 들어가면 반드시 핑거를 스친다(이전 구현의 실측 실패 양상).
        yaw_ok = _wrap_pi(yaw - final_yaw).abs() < self.entry_yaw_tol
        aligned = (x_err.abs() < self.entry_x_tol) & yaw_ok
        abort = x_err.abs() > self.entry_abort_x
        self._in_final = (self._in_final | aligned) & (~abort)

        # 조준점: 정렬 전이면 입구 앞 대기선까지만 간다(거기서 감속해 멈춰 정렬한다)
        gate_y = entrance_y - self.entry_standoff
        aim_y = torch.where(self._in_final, target[:, 1], torch.minimum(target[:, 1], gate_y))
        dx = berth_x - pos_xy[:, 0]
        dy = aim_y - pos_xy[:, 1]
        dist = torch.hypot(dx, dy)

        avoiding = torch.zeros_like(dist, dtype=torch.bool)
        if fls is not None:
            desired_yaw, avoiding = self._avoid(fls, yaw, desired_yaw, pos_xy, target)

        e_yaw = _wrap_pi(desired_yaw - yaw)

        # ── ② 제동거리 기반 감속 ──────────────────────────────────────────
        # **부호 있는 잔여거리**를 쓴다. 거리는 부호가 없으므로 목표를 지나쳐도
        # 계속 전진을 명령했고, 배가 벽을 뚫고 나간 뒤 제자리에서 회전했다.
        #
        # ★ 투영 축은 **경로(중심선) 방향**이다. 선수 방향이 아니다.
        #   선수축에 투영했더니 게걸음각이 커질 때 잔여거리가 0 으로 붕괴했다.
        #   실측: 대기선까지 1.36 m 남았는데 선수 134°, 목표 방위 224° 라
        #   along = 0.681 - 0.676 = +0.005 → 제어기가 "다 왔다"로 판단해 감속을 멈췄고,
        #   그대로 대기선을 0.94 m 지나쳐 선수 꼭짓점이 핑거 모서리를 스쳤다.
        #   유속과 싸우느라 게걸음각이 클수록 이 붕괴가 심해진다.
        along = dx * torch.cos(final_yaw) + dy * torch.sin(final_yaw)

        # v_ref = sqrt(2·a·along) — 남은 거리에서 정지할 수 있는 최대 속력.
        # 이전 구현은 v_ref = along×0.6 (상한 0.6) 이라 종방향 1 m 안에 들어와야
        # 감속이 시작됐고, 그 전에 핑거에 부딪혔다(실측 2.46 m 지점, 0.51 m/s 순항 중).
        v_brake = torch.sqrt(2.0 * self.brake_accel * along.clamp(min=0.0))
        v_max = torch.where(avoiding, torch.full_like(along, self.avoid_slow),
                            torch.full_like(along, self.approach_speed))
        v_ref = torch.where(along >= 0.0, torch.minimum(v_brake, v_max),
                            (along * 0.6).clamp(min=-0.30))

        # ★ 속도 오차도 **같은 축**에서 재야 한다.
        #   잔여거리만 경로축으로 바꾸고 속도는 선체 전진속도 nu[0] 를 그대로 썼더니
        #   게걸음 중 두 양이 다른 물리량이 되어 루프가 어긋났다(최종거리 16 m, 벽 충돌).
        #   경로 진행속도 = 속도벡터를 경로 방향에 투영한 값.
        psi_t = _wrap_pi(yaw - final_yaw)  # 경로 대비 선수 오차
        v_path = nu[:, 0] * torch.cos(psi_t) - nu[:, 1] * torch.sin(psi_t)
        # 정렬 전에는 속력을 0 으로 떨어뜨리지 않는다(위 v_min_transit 주석 참조).
        # 최종 진입 단계에서만 제동 프로파일이 0 까지 지배한다.
        v_ref = torch.where(self._in_final, v_ref,
                            torch.maximum(v_ref, torch.full_like(v_ref, self.v_min_transit)))
        e_dist = v_ref - v_path

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
