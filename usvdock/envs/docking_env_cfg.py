"""도킹 환경 설정.

Arm 정의 (실험 설계):
    A  LiDAR → 핑거 검출 → 버스 자세 추정 → PID   (고전 파이프라인, 학습 없음)
    B  LiDAR → RL
    C  LiDAR + FLS → RL

Mission 정의:
    1  핑거 2개 사이 주차 (수중 장애물 없음) — **통제군**
    2  1 + 수중 돌출 기초부

비교 축이 하나씩만 바뀐다:
    A vs B  정보 동일, 제어기만 다름  → 제어기의 가치
    B vs C  제어기 동일, 정보만 다름  → 센서 융합의 가치
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .. import blueboat_cfg as BB

# LiDAR 72 bin + 자기상태 8 + 직전행동 2 + 버스추정 특징 5(추측항법 유지)
_OBS_LIDAR = 72 + 8 + 2 + 5
# FLS 는 128 빔을 32 로 다운샘플 (정책 입력을 과하게 키우지 않는다)
_FLS_BINS = 32


@configclass
class DockingEnvCfg(DirectRLEnvCfg):
    """BlueBoat 도킹 환경."""

    # --- 실험 조건 ---
    mission: int = 2  # 1 또는 2
    use_fls: bool = True  # False = Arm B, True = Arm C
    # ── 유속 범위는 수조 사양이 아니라 **플랫폼의 물리 한계**로 정한다 ──────────
    # 도킹 자세는 벽에 수직(선수 90°)이고 수조 유속은 길이 방향이므로 **정횡(beam-on)**이다.
    # BlueBoat 는 횡추력이 없고 선수가 90° 로 고정되면 추력이 전부 종방향이라
    # 횡력을 만들 수단이 없다. 버틸 수 있는 것은 선체 횡항력뿐이다:
    #
    #     필요 횡력 |Y| = 60·v + 180·v²   vs   최대 추력 70 N
    #     → 정횡 유지 한계유속 = 0.479 m/s (0.93 knot)
    #
    # 이를 넘으면 **어떤 제어기로도** 도킹 자세를 유지할 수 없다(제어 문제가 아니라 물리).
    # 실측으로도 PID 성공률이 0.0 m/s 95.3% → 0.5 m/s 0.0% 로 절벽처럼 떨어진다.
    #
    # 수조 최대 3.4 kn(1.749 m/s)은 이 한계의 3.7배다. 수조 사양을 그대로 실험 축으로
    # 쓰면 안 되며, 2026-08 실기 실험에서도 0.9 kn 이상 올릴 이유가 없다.
    current_range: tuple[float, float] = (0.0, 0.45)  # 한계 0.479 m/s 바로 아래

    # --- 에피소드 ---
    episode_length_s: float = 40.0
    decimation: int = 2
    action_space: int = 2  # [좌현 추력, 우현 추력] 정규화
    observation_space: int = _OBS_LIDAR + _FLS_BINS
    state_space: int = 0

    # --- 시뮬레이션 ---
    # 물리 스텝 100 Hz, decimation 2 → 정책 50 Hz.
    # 도킹은 저속 정밀 제어라 50 Hz 면 충분하고, 40 초 에피소드가 2000 스텝이 된다.
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=2)

    # env_spacing 0 으로 모든 환경을 겹쳐 놓으면 PhysX GPU 가 죽는다.
    #   Warp CUDA error 700 (memset_device) + computeArticulationData: CUDA error 700
    # 동일 좌표에 강체가 겹쳐 생성되기 때문이다. 수조(35×20)보다 넉넉히 띄운다.
    # 좌표는 **env-local** 로 다루고, 시뮬레이터에 쓸 때만 env_origins 를 더한다.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1024,
        env_spacing=60.0,
        replicate_physics=True,  # 모든 환경의 USD 가 동일하다(무작위화는 torch 파라미터로)
    )

    # --- 선체 ---
    # 시각·충돌 모두 직육면체 근사. 실제 유체력은 검증된 3자유도 Fossen 모델이
    # 제공하므로 형상 메시가 물리에 관여하지 않는다.
    # (충돌 판정도 geometry.check_collision 의 해석식을 쓴다)
    robot: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Boat",
        spawn=sim_utils.CuboidCfg(
            size=(BB.LOA, BB.BEAM, 0.30),
            # kinematic_enabled 를 켜지 않는다.
            #   PhysX GPU 파이프라인에서 kinematic 강체는 별도 버퍼를 쓰는데
            #   write_root_pose_to_sim 과 맞물려 CUDA illegal memory access 를 냈다.
            #   중력만 끄고 일반 동적 강체로 두되, 매 스텝 자세와 속도를 덮어쓴다.
            #   (Isaac Lab 에서 운동학 구동을 하는 표준적 방법)
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            # collision_props 를 주지 않는다 — 충돌은 geometry.check_collision 의
            # 해석식으로 판정한다(프로펠러 최심점 기준). PhysX 접촉은 쓰지 않으므로
            # 콜라이더를 만들 이유가 없고, 만들면 GPU 브로드페이즈 부담만 는다.
            mass_props=sim_utils.MassPropertiesCfg(mass=BB.MASS),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # --- 보상 가중치 ---
    # 초기 설계는 학습이 전혀 되지 않았다(120 iter 동안 -105 → -131, 대부분 시간초과).
    # 원인: 도킹 보너스가 희소해 한 번도 못 받고, 그 사이 시간 페널티(-0.02×2000=-40)가
    #       전진 보상(총합이 초기거리 ~12 로 telescoping)을 압도했다.
    # 수정: 전진 보상을 키우고, **거리에 대한 조밀한 근접 보상**을 추가하고,
    #       시간 페널티를 줄인다. 근접 보상이 도킹 보너스로 가는 경사를 만들어 준다.
    # 2차 수정: 위 설계도 실패했다(400 iter, 성공률 0.0, final_dist 1.70 m).
    #   진단 지표가 원인을 정확히 짚었다 — 슬롯 입구에서 **맴돌기**가 최적이 되었다.
    #   exp(-dist/3) 근접보상은 1.7 m 에서도 스텝당 0.17 을 계속 준다.
    #   2000 스텝이면 +340. 반면 진입은 편측 여유 0.335 m 에 횡류 최대 1.75 m/s 라
    #   학습 초기 시도가 거의 다 충돌(-150)한다. 밖에 있는 편이 이득인 구조였다.
    #   → 근접보상 감쇠거리를 3.0 → 1.2 m 로 좁혀 입구에서 거의 안 주도록 하고,
    #     충돌 벌점을 낮춰 탐색을 덜 억제하고, 도킹 보너스를 키운다.
    rew_progress: float = 8.0  # 목표까지 거리 감소 (potential-based, 맴돌기 유인 없음)
    rew_proximity: float = 0.40  # exp(-dist/1.2) — 1.7 m 에서 0.098, 0.3 m 에서 0.31
    rew_proximity_scale: float = 1.2  # 감쇠거리 [m]
    rew_heading: float = 0.5  # 선수각 정렬 (근접할수록 가중)
    # 슬롯 **중심선 정렬** — 진입 여유가 편측 0.335 m 뿐이라 이것이 결정적이다.
    # 기존 보상에는 "목표까지 거리"와 "선수각"만 있고 "진입 전에 중심선에 올라타라"는
    # 신호가 없었다. 그래서 500 iter 에도 성공률이 13% 에 머물렀고 핑거 충돌이 26% 였다.
    # 벽에 가까울수록 가중을 키워, 멀리서는 자유롭게 접근하되 진입 직전엔 정렬하게 한다.
    #
    # 가중치 산정 (2.0 은 과했다 — Mean reward -1352):
    #   슬롯 입구에서 중심선 이탈 1 m 면 스텝당 -2, 2000 스텝이면 -4000.
    #   도킹 보너스 600 을 압도해 학습이 붕괴했다.
    #   0.15 로 낮추면: 이탈 0.5 m 로 2000 스텝 맴돌 때 -150 (보너스와 비교 가능한 규모),
    #   정렬(이탈 0.05 m)되면 -15 로 무시할 수준이 된다.
    #   또 이탈을 2 m 로 상한 처리해 먼 거리의 큰 값이 지배하지 않게 한다.
    # 최종 형태: **potential-based shaping** (Ng et al. 1999).
    #   단순 벌점(-w·cross·wall_prox)은 가중치를 2.0 → 0.15 로 낮춰도 실패했다
    #   (-349 → -43, 성공률 0). 가중치가 아니라 형태가 문제였다:
    #   벽에 가까울수록 벌점이 커지니 **접근 자체를 억제**한다. 배는 벽 근처로
    #   가야 배울 수 있는데 거기 가는 것을 벌하고 있었다.
    #
    #   퍼텐셜 Φ(s) = -w·cross·wall_prox 로 두고 r = γΦ(s') - Φ(s) 를 주면
    #   에피소드 합이 γ^T·Φ_T - Φ_0 로 유한하게 telescoping 되어 누적 편향이 없다.
    #   최적 정책을 바꾸지 않는 것이 증명되어 있어 "보상을 만져 결과를 만들었다"는
    #   비판에도 답할 수 있다.
    rew_crosstrack: float = 3.0
    crosstrack_cap: float = 2.0
    shaping_gamma: float = 0.995  # PPO 의 gamma 와 일치시킨다
    rew_dock_bonus: float = 600.0  # 도킹 성공 — 맴돌기 총합을 확실히 상회해야 한다
    pen_collision: float = -60.0  # 충돌. -150 은 탐색을 과하게 억눌렀다
    pen_effort: float = -0.002  # 추력 사용
    pen_rate: float = -0.01  # 추력 변화율 (실기 마모·전류 급변 방지)
    pen_time: float = -0.005  # 시간 (초기 -0.02 는 과했다)
    # 이동 **거리**에 값을 매긴다. 이것이 없으면 우회가 사실상 공짜라
    #   2 m 우회 비용 = 시간벌점 0.005 × 165스텝 ≈ 0.8  vs  충돌 -60
    # 즉 우회가 75배 싸서 정책이 최대한 크게 돌아간다(실측: PID 시간초과 92%).
    # 거리에 값을 매기면 우회가 직접 비용이 되고, 정책은 안전 경계를 따라
    # **최단 경로**로 붙어 지나가게 된다.
    pen_path: float = -1.0  # [보상/m]

    # --- 성공 판정 ---
    # 핑거 간격 1.6 m, 선폭 0.93 m → 현측 여유 0.335 m.
    # 횡방향 허용오차는 이보다 작아야 실제로 들어간 것이다.
    tol_lateral: float = 0.25  # x (벽을 따라)
    tol_longitudinal: float = 0.35  # y (벽으로 접근)
    tol_heading_deg: float = 12.0
    tol_speed: float = 0.20  # m/s
    # 0.5 초(25 스텝) 유지. 스치듯 지나가는 것은 막되 그 이상은 요구하지 않는다.
    #   부족구동 선박은 횡류 중 선수 90° 자세를 **물리적으로 유지할 수 없다**:
    #   평형 선수각이 유속과 나란한 방향뿐이라 Y 방향 평형이 존재하지 않는다.
    #   유속 0.45 m/s 에서 허용오차 0.25 m 이탈까지 0.69 초. 3 초 유지는 불가능하다.
    #   (실제로 슬립의 핑거와 계류줄이 그 역할을 대신한다)
    dock_hold_steps: int = 25

    # 시각화 전용: 도킹해도 에피소드를 끝내지 않는다.
    #   학습에서는 True(도달 즉시 종료). 영상에서는 False 로 두어 도킹 후에도 계속
    #   기록하고, 자세를 유지하려다 서서히 밀리는 실제 거동을 보여준다.
    terminate_on_dock: bool = True

    # --- 초기 배치 ---
    # 장거리 항주가 아니라 **종단 접근(terminal approach)** 문제로 한정한다.
    # 버스가 LiDAR 시야 안에 들어와 있는 상태에서 시작한다.
    init_dist_range: tuple[float, float] = (6.0, 12.0)
    init_bearing_deg: float = 35.0  # 버스 법선 기준 ± 범위
    init_yaw_err_deg: float = 40.0  # 선수각 오차 ± 범위

    # --- 유속 커리큘럼 ---
    # 진입 여유가 편측 0.335 m 인데 횡류가 최대 1.75 m/s 면 학습 초기에 진입을 배울 수 없다.
    # 무유속에서 진입 기동을 먼저 익히고 유속을 올린다. 표준적인 커리큘럼이며,
    # 외부 훅 없이 환경이 누적 스텝으로 자체 진행한다(rsl-rl 에 콜백을 넣지 않아도 된다).
    # 2048 env × 64 step/iter = 131,072 step/iter 이므로 300만 스텝은 23 iter 만에
    # 끝나 커리큘럼 구실을 못 했다(current_scale 이 즉시 1.0). 약 150 iter 로 늘린다.
    # 추측항법 주행거리계 잡음 (변위 대비 비율). DVL 속도 오차 1~2 % 수준.
    # 0 으로 두면 시뮬레이터의 정확한 변위를 쓰게 되어 실기에서 무너진다.
    odom_noise_frac: float = 0.02

    current_warmup_steps: int = 20_000_000
