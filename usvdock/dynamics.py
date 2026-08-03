"""3자유도 Fossen 조종 모델 — 수상선용.

marinelab 의 6자유도 수중 모델을 그대로 쓰지 않는 이유:
  marinelab/core/hydrodynamics.py:50 에 "No wave or surface effects" 라고 명시돼 있다.
  완전 몰수체(fully submerged) 가정이라 부력이 항상 상수이고, 자유표면 근처에서
  흘수에 따라 부력이 변하는 거동이 없다.

본 모델의 처리:
  수상선을 흘수면에 구속된 3자유도(surge u, sway v, yaw r) 조종 모델로 다룬다.
  heave/roll/pitch 는 정적 수조 도킹에서 관심 대상이 아니고, 자유표면 유체동역학을
  구현하지 않아도 되므로 시간과 위험을 동시에 줄인다.
  Fossen 의 3자유도 조종 모델은 도킹·DP(동적위치제어) 문헌의 표준 형태다.

운동방정식 (Fossen, Handbook of Marine Craft Hydrodynamics and Motion Control):

    M ν̇ + C(ν_r) ν_r + D(ν_r) ν_r = τ

    ν   = [u, v, r]ᵀ   선체고정 속도 (surge, sway, yaw rate)
    ν_r = ν - ν_c      **상대속도** — 유체력은 물에 대한 상대속도로 발생한다
    ν_c                유속을 선체좌표로 변환한 값 (yaw rate 성분은 0)
    τ   = B·[T_port, T_stbd]ᵀ   추진기 배분

PhysX 와의 역할 분담:
  PhysX 가 강체 관성(M_RB)과 강체 코리올리(C_RB)를 이미 적분한다.
  따라서 본 모듈은 **유체력만** 반환한다: 부가질량, 부가질량 코리올리, 감쇠.
  이것을 외력 렌치로 강체에 가한다 (marinelab 이 쓰는 방식과 동일).
"""

from __future__ import annotations

import torch

from . import blueboat_cfg as C


class SurfaceDynamics3DOF:
    """3자유도 Fossen 조종 모델. 배치(num_envs) 병렬."""

    def __init__(self, num_envs: int, device: str, cfg: C.HydroCfg | None = None, dt: float = 0.01):
        self.num_envs = num_envs
        self.device = device
        self.dt = dt
        self.cfg = cfg if cfg is not None else C.HydroCfg()

        h = self.cfg
        # 부가질량 행렬 M_A = -diag(X_udot, Y_vdot, N_rdot).
        # 계수가 음수로 정의되므로 부호를 뒤집으면 양수가 된다.
        self.m_a = torch.tensor(
            [-h.X_udot, -h.Y_vdot, -h.N_rdot], device=device, dtype=torch.float32
        )
        # 강체 질량/관성 (PhysX 가 담당하지만 코리올리 계산에 필요)
        self.m_rb = torch.tensor([C.MASS, C.MASS, C.I_ZZ], device=device, dtype=torch.float32)
        # 유효 질량 = 강체 + 부가 (코리올리 항에 쓰인다)
        self.m_eff = self.m_rb + self.m_a

        self.lin_damp = torch.tensor([-h.X_u, -h.Y_v, -h.N_r], device=device, dtype=torch.float32)
        self.quad_damp = torch.tensor(
            [-h.X_uu, -h.Y_vv, -h.N_rr], device=device, dtype=torch.float32
        )

        # 부가질량은 가속도에 비례한다. 명시적으로 다루면 불안정해질 수 있어
        # 직전 스텝 속도를 저장해 후진차분으로 가속도를 추정한다 (marinelab 과 동일한 접근).
        self._nu_prev = torch.zeros(num_envs, 3, device=device, dtype=torch.float32)
        self._initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

    # ------------------------------------------------------------------ 추진
    @staticmethod
    def allocate(thrust_port: torch.Tensor, thrust_stbd: torch.Tensor) -> torch.Tensor:
        """추진기 추력 → 선체좌표 일반화력 τ = [X, Y, N].

        Y 성분이 항상 0 이라는 점이 이 플랫폼의 본질이다(부족구동).
        횡방향으로 밀 수단이 없으므로 횡류 중 도킹은 crabbing 을 요구한다.
        """
        ly = C.THRUSTER_MOMENT_ARM
        X = thrust_port + thrust_stbd
        Y = torch.zeros_like(X)  # ← 부족구동. 이 0 이 문제의 난이도다.
        N = ly * (thrust_port - thrust_stbd)
        return torch.stack([X, Y, N], dim=-1)

    @staticmethod
    def actions_to_thrust(actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """정책 출력 [-1,1]² → 추력 [N]. 좌/우현 각각."""
        a = actions.clamp(-1.0, 1.0) * C.THRUST_MAX_PER_MOTOR
        return a[:, 0], a[:, 1]

    # ------------------------------------------------------------------ 유체력
    def fluid_wrench(self, nu: torch.Tensor, nu_current: torch.Tensor) -> torch.Tensor:
        """유체력만 반환 [N, N, N·m]. 강체 관성/코리올리는 PhysX 담당.

        Args:
            nu:         (N,3) 선체좌표 속도 [u, v, r]
            nu_current: (N,3) 선체좌표 유속 [u_c, v_c, 0]
        """
        nu_r = nu - nu_current  # 상대속도 — 유체력의 기준

        # --- 감쇠 D(ν_r) ν_r : 선형 + 2차 ---
        damp = -(self.lin_damp + self.quad_damp * nu_r.abs()) * nu_r

        # --- 부가질량 코리올리 C_A(ν_r) ν_r ---
        # 3자유도 형태. 선회 시 횡력과 회두 모멘트의 연성을 만든다.
        u_r, v_r, r = nu_r[:, 0], nu_r[:, 1], nu_r[:, 2]
        m11, m22 = self.m_eff[0], self.m_eff[1]
        cor = torch.stack(
            [
                m22 * v_r * r,
                -m11 * u_r * r,
                (m11 - m22) * u_r * v_r,
            ],
            dim=-1,
        )

        # --- 부가질량 -M_A ν̇ ---
        # 직전 속도와의 후진차분으로 가속도 추정. 첫 스텝은 0 으로 둔다.
        acc = torch.where(
            self._initialized.unsqueeze(-1),
            (nu - self._nu_prev) / self.dt,
            torch.zeros_like(nu),
        )
        # 폭주 방지 클램프. 명시적 부가질량은 M_A/M_RB 비가 클 때 발산할 수 있다.
        acc = acc.clamp(-50.0, 50.0)
        added = -self.m_a * acc

        self._nu_prev = nu.clone()
        self._initialized[:] = True

        return damp + cor + added

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        self._nu_prev[env_ids] = 0.0
        self._initialized[env_ids] = False


# ---------------------------------------------------------------------------
# 검증: 제조사 공식 최대속력 재현
# ---------------------------------------------------------------------------
def verify_max_speed(verbose: bool = True) -> float:
    """양현 최대 추력에서 정상상태 속력을 구해 [SPEC] 3.0 m/s 와 비교한다.

    X_uu 를 이 값에서 역산했으므로, 되돌아 나오는지 확인하는 왕복 검증이다.
    선형 감쇠 X_u 가 함께 작용하므로 3.0 보다 약간 낮게 나오는 것이 정상이다.
    """
    dev = "cpu"
    dyn = SurfaceDynamics3DOF(1, dev, dt=0.01)
    nu = torch.zeros(1, 3, device=dev)
    zero_cur = torch.zeros(1, 3, device=dev)
    tau = dyn.allocate(
        torch.tensor([C.THRUST_MAX_PER_MOTOR]), torch.tensor([C.THRUST_MAX_PER_MOTOR])
    )

    dt = 0.01
    m_total = dyn.m_rb + dyn.m_a  # 전진 가속에는 부가질량이 함께 작용
    for _ in range(20000):  # 200 초
        f = dyn.fluid_wrench(nu, zero_cur)
        nu = nu + dt * (tau + f) / m_total

    u_ss = float(nu[0, 0])
    if verbose:
        print(f"양현 최대추력 {2*C.THRUST_MAX_PER_MOTOR:.0f} N")
        print(f"정상상태 전진속력  {u_ss:.3f} m/s   ([SPEC] {C.MAX_SPEED} m/s)")
        print(f"오차 {100*(u_ss-C.MAX_SPEED)/C.MAX_SPEED:+.1f} %")
        print()
        print("참고: 유효질량(강체+부가)")
        print(f"  surge {float(m_total[0,0] if m_total.ndim>1 else m_total[0]):.2f} kg")
    return u_ss


if __name__ == "__main__":
    verify_max_speed()
