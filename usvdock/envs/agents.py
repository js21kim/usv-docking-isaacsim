"""rsl-rl PPO 설정.

**stock rsl-rl 3.1.2 를 그대로 쓴다.** constrained-albc 의 ConstraintTRPO 를 쓰지 않는
이유는 두 가지다: (1) 그쪽은 연구실 동료의 비공개 연구 코드라 의존을 만들고 싶지 않고,
(2) 영어 10분 발표에서 표준 PPO 가 설명 비용이 훨씬 싸다.
본 연구의 기여는 알고리즘이 아니라 **센서 구성의 상보성**이므로 제어기는 표준이 낫다.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class DockingPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """도킹용 PPO.

    관측이 LiDAR 72 + FLS 32 로 커서 은닉층을 BlueROV(128-128-64)보다 넓게 잡는다.
    도킹은 종단에서 정밀도가 필요하므로 horizon 을 길게(64) 둔다.
    """

    num_steps_per_env = 64
    max_iterations = 600
    # 10 iter 마다 저장한다. 발표용 "학습 진행 4컷"을 **데이터로 골라야** 하기 때문이다:
    #   완전 실패 → 접근하다 충돌 → 도달했으나 감속 실패 → 감속·주차 성공
    # 이 전환점이 어디인지 미리 알 수 없다. 100 간격이면 초반 전환을 통째로 놓친다.
    # 3000 iter → 300 체크포인트 × 6 MB ≈ 1.8 GB/run (디스크 947 GB 라 무관).
    save_interval = 10
    experiment_name = "usv_docking"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.995,  # 도킹은 장기 신용할당이 필요하다
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
