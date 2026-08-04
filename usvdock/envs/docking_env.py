"""BlueBoat 도킹 환경 (Isaac Lab DirectRLEnv).

── 동역학 처리 방식과 그 이유 ────────────────────────────────────────────────
선체를 PhysX 강체로 두고 외력을 가하는 대신, **검증된 3자유도 Fossen 모델을
torch 로 적분하고 그 자세를 시뮬레이터에 기록(kinematic)** 한다.

이유:
  (1) 수상선을 PhysX 로 띄우려면 자유표면 부력·복원력을 튜닝해야 하는데,
      정적 수조 도킹에서 heave/roll/pitch 는 관심 대상이 아니다.
      튜닝 위험만 크고 얻는 것이 없다.
  (2) 3자유도 Fossen 모델은 **제조사 공식 최대속력 3.0 m/s 를 재현하도록 보정**하고
      단위시험으로 확인했다(dynamics.verify_max_speed). 물리 근거가 명확하다.
  (3) 충돌은 PhysX 접촉이 아니라 geometry.check_collision 의 해석식으로 판정한다.
      프로펠러 최심점(0.272 m) 기준 판정이 필요한데, 이는 직육면체 근사 강체의
      접촉으로는 표현할 수 없다.

Isaac Sim/Isaac Lab 이 담당하는 것: 장면 구성, 렌더링(발표용 영상), 환경 API,
병렬 실행, RL 스택(rsl-rl). 유체동역학만 본 프로젝트가 제공한다.
marinelab 도 유체력을 직접 계산해 PhysX 에 가하는 구조이므로 결이 같다.

── 관측 설계 ────────────────────────────────────────────────────────────────
정책에 **목표 위치를 주지 않는다.** 버스 위치는 LiDAR 로 알아내야 한다.
목표를 그냥 주면 인지 문제가 사라져 Arm A/B/C 비교가 무의미해진다.
보상 계산에는 목표를 쓴다(학습 시 특권 정보 사용은 표준적 관행).
"""

from __future__ import annotations

import math

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import DirectRLEnv

from .. import blueboat_cfg as BB
from .. import geometry as G
from .. import perception as PC
from ..dynamics import SurfaceDynamics3DOF
from .docking_env_cfg import _FLS_BINS, DockingEnvCfg


def _wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DockingEnv(DirectRLEnv):
    cfg: DockingEnvCfg

    def __init__(self, cfg: DockingEnvCfg, render_mode: str | None = None, **kw):
        super().__init__(cfg, render_mode, **kw)
        N, dev = self.num_envs, self.device

        self._dyn = SurfaceDynamics3DOF(N, dev, dt=self.step_dt)

        # 상태: eta(x,y,yaw) 세계좌표, nu(u,v,r) 선체좌표
        self._eta = torch.zeros(N, 3, device=dev)
        self._nu = torch.zeros(N, 3, device=dev)

        # 추진기 1차 지연 (M200 응답). 실제 추진기는 즉시 반응하지 않는다.
        self._thrust = torch.zeros(N, 2, device=dev)
        self._thrust_tau = 0.15  # [EST] 시정수 [s]

        self._prev_action = torch.zeros(N, 2, device=dev)
        self._scene_p: G.SceneParams | None = None
        self._target = torch.zeros(N, 3, device=dev)
        self._prev_dist = torch.zeros(N, device=dev)
        self._prev_phi = torch.zeros(N, device=dev)
        self._hold = torch.zeros(N, dtype=torch.long, device=dev)
        self._docked = torch.zeros(N, dtype=torch.bool, device=dev)
        self._collided = torch.zeros(N, dtype=torch.bool, device=dev)
        self._entangled = torch.zeros(N, dtype=torch.bool, device=dev)
        self._min_clear = torch.full((N,), 1e3, device=dev)
        self._path_len = torch.zeros(N, device=dev)
        self._finger = torch.zeros(N, dtype=torch.bool, device=dev)
        self._wall = torch.zeros(N, dtype=torch.bool, device=dev)
        # 종료 사유 코드. **_reset_idx 에서 지우지 않는다.**
        #   DirectRLEnv.step() 은 dones 계산 직후 내부에서 자동 리셋을 돌린다.
        #   그래서 step() 이 반환된 뒤 _docked/_entangled/... 를 읽으면 이미 초기화된 값이다
        #   (롤아웃에서 충돌한 에피소드가 전부 timeout 으로 찍혀 발견).
        #   0=없음 1=도킹 2=폐어구감김 3=핑거 4=벽
        self._outcome_code = torch.zeros(N, dtype=torch.long, device=dev)

        self._env_steps = 0  # 유속 커리큘럼 진행도
        self._sensor_cache = None  # 스텝당 센서 캐시
        self._sensor_key = -1
        # ── 버스 추정 유지 (최소 기억) ────────────────────────────────────
        # 단일 프레임 검출만 쓰면 **종단에서 눈이 감긴다**: 배가 버스 옆으로 붙으면
        # 가까운 핑거가 먼 핑거를 가려 한쪽만 보이고, "핑거 쌍" 조건이 깨진다.
        # 실측: 벽까지 2 m 안쪽에서 횡오프셋 1 m 만 있어도 검출 실패.
        # 실제로 정책이 목표 1.9 m 앞에서 검출을 잃고 40초를 배회했다(검출률 23.4%).
        #
        # → 마지막 유효 추정을 **자기 운동으로 추측항법**하며 유지하고,
        #   경과 시간을 정책에 함께 준다. 실무에서 표준적인 처리이며,
        #   필요한 자기 운동 정보는 IMU/DVL 로 실제 관측 가능한 양이다.
        #   (전역 지도나 SLAM 이 아니다 — 대상 하나의 상대 위치를 끌고 갈 뿐이다)
        self._berth_rel = torch.zeros(N, 2, device=dev)  # 세계축 정렬, 선체 중심
        self._berth_age = torch.full((N,), 1e3, device=dev)  # 마지막 검출 후 경과 [s]
        self._berth_key = -1  # 스텝당 1회만 갱신 (센서 캐시와 같은 방식)
        self._last_dxy = torch.zeros(N, 2, device=dev)  # 직전 스텝 세계 변위
        self._resample_scene(torch.arange(N, device=dev))

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        import isaaclab.sim as sim_utils

        self._boat = RigidObject(self.cfg.robot)
        self.scene.rigid_objects["boat"] = self._boat
        # 정적 장면은 **복제 전에** env_0 아래에 만든다. clone_environments 가 복제한다.
        self._spawn_static_scene("/World/envs/env_0")
        self.scene.clone_environments(copy_from_source=False)

        light = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.85, 0.95))
        light.func("/World/Light", light)

    def _spawn_static_scene(self, root: str):
        """수조·버스 시각화. 물리에 관여하지 않는 순수 시각 프림이다.

        복제 전에 env_0 아래에 만들어야 clone_environments 가 모든 환경에 복사한다.
        """
        import isaaclab.sim as sim_utils

        t, b = BB.TankCfg(), BB.BerthCfg()

        def box(path, size, pos, color):
            cfg = sim_utils.CuboidCfg(
                size=size,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                collision_props=None,
            )
            cfg.func(path, cfg, translation=pos)

        # 수면 (시각용. LiDAR 레이캐스트 대상이 아님 — geometry.py 주석 참조)
        box(f"{root}/Water", (t.length, t.width, 0.02), (0.0, 0.0, -0.01), (0.10, 0.35, 0.50))
        # 벽 4면
        th = 0.3
        box(f"{root}/WallN", (t.length, th, 2.0), (0.0, t.width / 2 + th / 2, 0.5), (0.6, 0.6, 0.6))
        box(f"{root}/WallS", (t.length, th, 2.0), (0.0, -t.width / 2 - th / 2, 0.5), (0.5, 0.5, 0.5))
        box(f"{root}/WallE", (th, t.width, 2.0), (t.length / 2 + th / 2, 0.0, 0.5), (0.5, 0.5, 0.5))
        box(f"{root}/WallW", (th, t.width, 2.0), (-t.length / 2 - th / 2, 0.0, 0.5), (0.5, 0.5, 0.5))
        # 핑거 2개 (대표 위치 x=0. 실제 버스 x 는 에피소드마다 다르므로 시각화는 참고용)
        off = b.pile_gap / 2 + b.pile_diameter / 2
        for s, nm in ((-1, "P"), (1, "S")):
            box(
                f"{root}/Finger{nm}",
                (b.pile_diameter, b.pile_length, 0.7),
                (s * off, G.WALL_Y - b.pile_length / 2, 0.15),
                (0.85, 0.75, 0.35),
            )

    # ------------------------------------------------------------------ step
    def _pre_physics_step(self, actions: torch.Tensor):
        self._prev_action = self._actions.clone() if hasattr(self, "_actions") else actions.clone()
        self._actions = actions.clamp(-1.0, 1.0)
        # 추진기 1차 지연
        cmd = self._actions * BB.THRUST_MAX_PER_MOTOR
        a = self.step_dt / (self._thrust_tau + self.step_dt)
        self._thrust = self._thrust + a * (cmd - self._thrust)

    def _apply_action(self):
        """3자유도 Fossen 적분 후 자세를 시뮬레이터에 기록."""
        p = self._scene_p
        # 유속을 선체좌표로 (수조 유속은 세계 x 축)
        c, s = torch.cos(self._eta[:, 2]), torch.sin(self._eta[:, 2])
        nu_c = torch.stack(
            [p.current_u * c, -p.current_u * s, torch.zeros_like(c)], dim=-1
        )

        tau = self._dyn.allocate(self._thrust[:, 0], self._thrust[:, 1])
        f = self._dyn.fluid_wrench(self._nu, nu_c)
        m_eff = self._dyn.m_rb + self._dyn.m_a
        self._nu = self._nu + self.step_dt * (tau + f) / m_eff
        self._nu = self._nu.clamp(-5.0, 5.0)

        # 선체 → 세계
        c, s = torch.cos(self._eta[:, 2]), torch.sin(self._eta[:, 2])
        vx = self._nu[:, 0] * c - self._nu[:, 1] * s
        vy = self._nu[:, 0] * s + self._nu[:, 1] * c
        dx_w = self.step_dt * vx
        dy_w = self.step_dt * vy
        self._last_dxy = torch.stack([dx_w, dy_w], dim=-1)  # 추측항법용 세계 변위
        self._eta[:, 0] += dx_w
        self._eta[:, 1] += dy_w
        self._eta[:, 2] = _wrap_pi(self._eta[:, 2] + self.step_dt * self._nu[:, 2])

        self._write_pose()

    def _crosstrack_potential(self, pos: torch.Tensor) -> torch.Tensor:
        """중심선 이탈 퍼텐셜 (양수). 벽에 가까울수록 크다."""
        c = self.cfg
        cross = (pos[:, 0] - self._target[:, 0]).abs().clamp(max=c.crosstrack_cap)
        wall_prox = torch.exp(
            -(G.WALL_Y - G.BerthCfg_pile_len - pos[:, 1]).clamp(min=0.0) / 2.0
        )
        return cross * wall_prox * c.rew_crosstrack

    def _write_pose(self):
        """env-local 자세를 시뮬레이터의 세계좌표로 변환해 기록.

        _eta 는 **env-local** 이다(각 환경이 자기 수조 원점을 기준으로 함).
        시뮬레이터에는 env_origins 를 더한 세계좌표를 줘야 한다.
        geometry.py 의 모든 계산도 env-local 이므로 그대로 쓰면 된다.
        """
        N = self.num_envs
        origins = self.scene.env_origins  # (N,3)
        pose = torch.zeros(N, 7, device=self.device)
        pose[:, 0] = self._eta[:, 0] + origins[:, 0]
        pose[:, 1] = self._eta[:, 1] + origins[:, 1]
        pose[:, 2] = origins[:, 2]
        half = self._eta[:, 2] * 0.5
        pose[:, 3] = torch.cos(half)  # w
        pose[:, 6] = torch.sin(half)  # z
        self._boat.write_root_pose_to_sim(pose)
        # 속도도 0 으로 덮어쓴다. 안 그러면 PhysX 가 남은 속도로 자체 적분을 이어가
        # 우리가 적분한 자세와 어긋난다. 운동은 전적으로 Fossen 모델이 결정한다.
        self._boat.write_root_velocity_to_sim(torch.zeros(N, 6, device=self.device))

    # ------------------------------------------------------------------ 센서
    def sensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """스텝당 한 번만 스캔하고 캐시한다. (고해상도 LiDAR, FLS)

        같은 스텝에서 관측·버스검출·PID 회피가 각자 스캔을 다시 뜨면 낭비가 크다.
        평가에서 스텝당 LiDAR 스캔이 3회, FLS 가 2회 돌고 있었다.

        72빈 정책 입력은 360빈에서 5칸씩 뽑으면 **정확히 같은 값**이다:
            72빈[i] 의 방위 = i·2π/72 = (5i)·2π/360 = 360빈[5i] 의 방위
        """
        key = int(self.common_step_counter)
        if self._sensor_key == key and self._sensor_cache is not None:
            return self._sensor_cache
        s = BB.SensorMountCfg()
        pos, yaw = self._eta[:, :2], self._eta[:, 2]
        lid = G.lidar_scan(pos, yaw, self._scene_p, n_bins=s.lidar_det_h_bins, mount=s)
        fls = G.fls_scan(pos, yaw, self._scene_p)
        self._sensor_cache = (lid, fls)
        self._sensor_key = key
        return self._sensor_cache

    # ------------------------------------------------------------------ obs
    def _get_observations(self) -> dict:
        p = self._scene_p
        pos = self._eta[:, :2]
        yaw = self._eta[:, 2]

        s_cfg = BB.SensorMountCfg()
        lid360, fls_full = self.sensors()
        step = s_cfg.lidar_det_h_bins // s_cfg.lidar_obs_h_bins  # 360//72 = 5
        lidar = lid360[:, ::step]
        ego = torch.stack(
            [
                self._nu[:, 0] / BB.MAX_SPEED,
                self._nu[:, 1] / BB.MAX_SPEED,
                self._nu[:, 2] / 2.0,
                torch.cos(yaw),
                torch.sin(yaw),
                self._thrust[:, 0] / BB.THRUST_MAX_PER_MOTOR,
                self._thrust[:, 1] / BB.THRUST_MAX_PER_MOTOR,
                self.episode_length_buf.float() / self.max_episode_length,
            ],
            dim=-1,
        )
        # 세 방식이 공유하는 인지 프론트엔드. PID 도 같은 검출기를 쓴다.
        # 정답 좌표가 아니라 LiDAR 스캔에서 추정한 값이므로 공정성이 유지된다.
        berth = self._berth_features()
        parts = [ego, self._prev_action, berth, lidar]

        if self.cfg.use_fls:
            fls = fls_full.view(self.num_envs, _FLS_BINS, -1).min(dim=-1).values  # 최근접 보존
        else:
            # Arm B: FLS 없음. 정책 입력 크기를 맞추기 위해 상수 1.0(무반사)로 채운다.
            # 값이 항상 같으므로 정보가 되지 않는다.
            fls = torch.ones(self.num_envs, _FLS_BINS, device=self.device)
        parts.append(fls)

        return {"policy": torch.cat(parts, dim=-1)}

    def update_berth_estimate(self) -> tuple[torch.Tensor, torch.Tensor]:
        """버스 추정을 갱신한다. 검출되면 새 값, 아니면 추측항법으로 끌고 간다.

        ★ **스텝당 한 번만** 갱신한다. PID 경로에서는 정책이 한 번, 환경의
          _get_observations 가 또 한 번 불러 추측항법이 **이중 적용**된다.
          (RL 경로는 관측에서만 불려 1회 — 두 방식이 서로 다른 추정을 받게 된다)
          센서 캐시와 같은 방식으로 스텝 카운터에 걸어 멱등하게 만든다.

        Returns: (추정 (N,2) [상대x, 벽거리], 경과시간 (N,))
        """
        key = int(self.common_step_counter)
        if self._berth_key == key:
            return self._berth_rel, self._berth_age
        self._berth_key = key

        lid360, _ = self.sensors()
        pos, yaw = self._eta[:, :2], self._eta[:, 2]
        rel_x, wall_d, ok = PC.detect_berth(pos, yaw, self._scene_p, scan=lid360)

        # 배가 움직인 만큼 상대 위치를 당긴다(세계축 정렬 좌표이므로 단순 뺄셈).
        #
        # ★ 시뮬레이터의 정확한 변위를 그대로 쓰면 **공짜 점심**이다. 수면을 LiDAR
        #   대상에서 뺀 것과 같은 이유로, 실기 DVL/IMU 수준의 오차를 섞는다.
        #   DVL 속도 오차는 보통 측정치의 1~2 % → 변위에 비례 잡음으로 준다.
        #   이 잡음이 누적되므로 오래 끌수록 추정이 실제로 나빠지고,
        #   정책은 "가끔은 다시 봐야 한다"를 배울 수밖에 없다.
        noisy = self._last_dxy * (
            1.0 + self.cfg.odom_noise_frac * torch.randn_like(self._last_dxy)
        )
        self._berth_rel = self._berth_rel - noisy
        fresh = torch.stack([rel_x, wall_d], dim=-1)
        self._berth_rel = torch.where(ok.unsqueeze(-1), fresh, self._berth_rel)
        self._berth_age = torch.where(ok, torch.zeros_like(self._berth_age),
                                      self._berth_age + self.step_dt)
        return self._berth_rel, self._berth_age

    def _berth_features(self) -> torch.Tensor:
        """정책 입력 (N,5): 선체좌표 상대위치 2 + 선수각오차 2 + 추정 신선도 1."""
        rel, age = self.update_berth_estimate()
        yaw = self._eta[:, 2]
        ty_off = 0.50 + BB.LOA / 2
        dx, dy = rel[:, 0], rel[:, 1] - ty_off
        c, s = torch.cos(yaw), torch.sin(yaw)
        bx = dx * c + dy * s
        by = -dx * s + dy * c
        yaw_err = _wrap_pi(math.pi / 2 - yaw)
        # 신선도: 방금 검출 1.0 → 오래될수록 0. 정책이 추정을 얼마나 믿을지 판단한다.
        fresh = torch.exp(-age / 2.0)
        return torch.stack(
            [bx / 20.0, by / 20.0, torch.cos(yaw_err), torch.sin(yaw_err), fresh], dim=-1
        )

    # ------------------------------------------------------------------ reward
    def _get_rewards(self) -> torch.Tensor:
        c = self.cfg
        pos, yaw = self._eta[:, :2], self._eta[:, 2]
        dist = torch.linalg.norm(pos - self._target[:, :2], dim=-1)

        progress = (self._prev_dist - dist) * c.rew_progress
        self._prev_dist = dist
        # 조밀 근접 보상 — 희소한 도킹 보너스로 가는 경사를 만든다.
        proximity = torch.exp(-dist / c.rew_proximity_scale) * c.rew_proximity

        # 중심선 정렬 — potential-based shaping.
        # Φ(s) = -w·cross·wall_prox,  r = γΦ(s') - Φ(s)
        # 진입이 임박할수록(벽에 가까울수록) 중심선 이탈의 퍼텐셜이 깊어지므로
        # "정렬하며 접근"이 이득이 된다. 단순 벌점과 달리 접근 자체를 벌하지 않는다.
        phi = -self._crosstrack_potential(pos)
        crosstrack = c.shaping_gamma * phi - self._prev_phi
        self._prev_phi = phi

        yaw_err = _wrap_pi(yaw - self._target[:, 2]).abs()
        # 멀리서는 선수각이 자유로워야 접근이 쉽다. 가까울수록 정렬 가중을 올린다.
        closeness = torch.exp(-dist / 2.0)
        heading = -yaw_err * closeness * c.rew_heading

        # 이동 거리 벌점 — 우회를 직접 비용으로 만든다
        step_len = torch.linalg.norm(self._nu[:, :2], dim=-1) * self.step_dt
        path = step_len * c.pen_path
        self._path_len = self._path_len + step_len

        effort = self._actions.pow(2).sum(-1) * c.pen_effort
        rate = (self._actions - self._prev_action).pow(2).sum(-1) * c.pen_rate

        col = self._collided.float() * c.pen_collision
        dock = self._docked.float() * c.rew_dock_bonus

        return progress + proximity + crosstrack + heading + path + effort + rate + col + dock + c.pen_time

    # ------------------------------------------------------------------ dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        p = self._scene_p
        pos, yaw = self._eta[:, :2], self._eta[:, 2]

        col = G.check_collision(pos, yaw, p)
        # 충돌 유형을 따로 보관한다. 평가에서 "무엇에 부딪혔는지"가 핵심 지표이며,
        # 특히 프로펠러 충돌률이 미션2에서 FLS 효과를 보여주는 숫자다.
        self._entangled = col["entangled"]  # 폐어구 안전여유 침범
        # 에피소드 중 최소 이격거리를 추적한다.
        # "학습될수록 아슬아슬하게 지나간다"를 보이려면 이 값이 필요하다.
        # 규정 안전여유(0.30 m)로 **위에서 수렴**하는 곡선이 결과물이다.
        self._min_clear = torch.minimum(self._min_clear, col["clearance"])
        self._finger = col["finger_hit"]
        self._wall = col["wall_hit"]
        self._collided = col["any"]

        err = pos - self._target[:, :2]
        yaw_err = _wrap_pi(yaw - self._target[:, 2]).abs()
        speed = torch.linalg.norm(self._nu[:, :2], dim=-1)
        ok = (
            (err[:, 0].abs() < c.tol_lateral)
            & (err[:, 1].abs() < c.tol_longitudinal)
            & (yaw_err < math.radians(c.tol_heading_deg))
            & (speed < c.tol_speed)
        )
        self._hold = torch.where(ok, self._hold + 1, torch.zeros_like(self._hold))
        self._docked = self._hold >= c.dock_hold_steps

        # 시각화 실행에서는 도킹으로 끝내지 않는다(cfg.terminate_on_dock=False).
        # 도킹 후 거동까지 보여주기 위한 것이며 학습에는 영향이 없다.
        terminated = (self._docked & c.terminate_on_dock) | self._collided
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # 진단 지표를 학습 로그로 내보낸다.
        # 보상값만으로는 "도킹해서 끝났는지 충돌해서 끝났는지" 구분할 수 없다.
        # 이 지표가 없으면 에피소드 길이가 줄어든 것을 개선으로 오해하게 된다.
        # 종료 사유를 리셋에 살아남는 버퍼에 기록한다 (위 주석 참조)
        code = torch.zeros_like(self._outcome_code)
        code = torch.where(self._wall, torch.full_like(code, 4), code)
        code = torch.where(self._finger, torch.full_like(code, 3), code)
        code = torch.where(self._entangled, torch.full_like(code, 2), code)
        code = torch.where(self._docked, torch.full_like(code, 1), code)
        self._outcome_code = torch.where(terminated, code, self._outcome_code)

        self._env_steps += self.num_envs
        done = terminated | truncated
        n = done.sum().clamp(min=1).float()
        self.extras["log"] = {
            "success_rate": (self._docked & done).sum() / n,
            "collision_rate": (self._collided & done).sum() / n,
            "timeout_rate": (truncated & ~terminated).sum() / n,
            "entangle_rate": (col["entangled"] & done).sum() / n,
            "finger_hit_rate": (col["finger_hit"] & done).sum() / n,
            "wall_hit_rate": (col["wall_hit"] & done).sum() / n,
            "current_scale": torch.tensor(self._current_scale(), device=self.device),
            # ── 이격거리 지표 ────────────────────────────────────────────────
            # 처음에는 "성공 에피소드의 최소 이격거리"를 내보냈는데 값이 무의미했다:
            #   rsl-rl 은 매 스텝의 log 값을 평균낸다. 성공 에피소드가 없는 스텝은
            #   0 을 기여하므로, 성공률 4% 에서는 대부분이 0 이 되어 희석된다.
            #   실제로 path_len 이 0.65 m 로 찍혔다 — 6~12 m 를 이동하는데 불가능한 값이다.
            # → **매 스텝 well-defined 한 값**으로 바꾼다: 현재 이격거리의 전체 평균.
            #   비활성 그물(1e6)이 지배하지 않도록 5 m 로 상한 처리한다.
            #   정확한 수치는 학습 로그가 아니라 eval 에서 체크포인트별로 측정한다.
            "clearance_now": col["clearance"].clamp(max=5.0).mean(),
            "close_rate": (col["clearance"] < 1.0).float().mean(),
            "final_dist": torch.linalg.norm(err, dim=-1)[done].mean() if done.any() else torch.zeros((), device=self.device),
        }
        return terminated, truncated

    # ------------------------------------------------------------------ reset
    def _current_scale(self) -> float:
        """유속 커리큘럼: 누적 스텝에 따라 0 → 1 로 선형 증가.

        ★ warmup <= 0 이면 커리큘럼을 끄고 항상 1.0 을 쓴다(평가용).
          평가에서 환경을 새로 만들면 _env_steps=0 이라 유속이 0 으로 강제된다.
          그러면 학습보다 훨씬 쉬운 조건에서 재게 되어 성능이 과대평가된다.
          (실제로 이 버그로 롤아웃 유속이 전부 0.00 으로 나왔다)
        """
        w = self.cfg.current_warmup_steps
        if w <= 0:
            return 1.0
        return min(1.0, self._env_steps / w)

    def _resample_scene(self, ids: torch.Tensor):
        n = len(ids)
        k = self._current_scale()
        lo, hi = self.cfg.current_range
        new = G.sample_scene(n, self.device, self.cfg.mission, (lo * k, hi * k))
        if self._scene_p is None:
            self._scene_p = G.sample_scene(
                self.num_envs, self.device, self.cfg.mission, self.cfg.current_range
            )
        for k in new.__dataclass_fields__:
            getattr(self._scene_p, k)[ids] = getattr(new, k)

    def _reset_idx(self, env_ids):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._boat._ALL_INDICES
        super()._reset_idx(env_ids)
        n = len(env_ids)
        dev = self.device
        c = self.cfg

        self._resample_scene(env_ids)
        tgt = G.docking_target(self._scene_p)
        self._target[env_ids] = tgt[env_ids]

        # 종단 접근 배치: 버스 법선(-y 방향) 기준 ± 방위각, 거리 6~12 m
        d = torch.rand(n, device=dev) * (c.init_dist_range[1] - c.init_dist_range[0]) + c.init_dist_range[0]
        bear = (torch.rand(n, device=dev) * 2 - 1) * math.radians(c.init_bearing_deg)
        bx = self._scene_p.berth_x[env_ids]
        self._eta[env_ids, 0] = bx + d * torch.sin(bear)
        self._eta[env_ids, 1] = G.WALL_Y - 2.5 - d * torch.cos(bear)
        # 선수는 대체로 버스 쪽(+y = pi/2) 이되 오차를 준다
        self._eta[env_ids, 2] = math.pi / 2 + (torch.rand(n, device=dev) * 2 - 1) * math.radians(
            c.init_yaw_err_deg
        )

        self._nu[env_ids] = 0.0
        self._thrust[env_ids] = 0.0
        self._prev_action[env_ids] = 0.0
        self._berth_rel[env_ids] = 0.0
        self._berth_age[env_ids] = 1e3
        self._last_dxy[env_ids] = 0.0
        self._berth_key = -1
        self._hold[env_ids] = 0
        self._docked[env_ids] = False
        self._collided[env_ids] = False
        self._entangled[env_ids] = False
        self._min_clear[env_ids] = 1e3
        self._path_len[env_ids] = 0.0
        self._finger[env_ids] = False
        self._wall[env_ids] = False
        self._dyn.reset_idx(env_ids)

        self._prev_dist[env_ids] = torch.linalg.norm(
            self._eta[env_ids, :2] - self._target[env_ids, :2], dim=-1
        )
        # 리셋 시 퍼텐셜을 현재 상태로 초기화한다. 안 하면 첫 스텝에 가짜 보상이 생긴다.
        self._prev_phi[env_ids] = -self._crosstrack_potential(self._eta[:, :2])[env_ids]
        self._write_pose()
