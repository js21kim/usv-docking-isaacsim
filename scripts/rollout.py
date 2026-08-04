"""체크포인트 하나로 에피소드를 굴려 **궤적 전체를 기록**한다.

용도:
  (1) 발표용 "학습 진행 4컷" 영상 — 여러 체크포인트를 같은 장면에서 굴려 비교
  (2) 사후 분석 — 최종 속력, 감속 시점 등 학습 로그에 없는 값을 여기서 얻는다
      (학습 4회의 조건을 동일하게 유지하기 위해 학습 중에는 지표를 추가하지 않았다)

★ 장면을 강제로 고정한다
  체크포인트마다 정책이 다르면 에피소드 길이가 달라지고, 그러면 리셋 순서가 달라져
  같은 seed 를 줘도 다른 장면이 나온다. 4컷을 나란히 놓으려면 **완전히 같은 장면**이어야
  하므로, 리셋 직후 장면 파라미터와 초기 자세를 직접 덮어쓴다.

실행 (컨테이너 안):
    python scripts/rollout.py --task Isaac-USVDock-M2-Fusion-v0 \
        --checkpoint logs/.../model_500.pt --scene 0 --out traj/it500.npz
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="궤적 기록 롤아웃")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--checkpoint", type=str, default=None, help="없으면 PID(Arm A)로 실행")
parser.add_argument("--scene", type=int, default=0,
                    help="고정 장면 번호 (0~4). -1 이면 고정하지 않고 환경의 무작위 장면을 쓴다")
parser.add_argument("--seed", type=int, default=12345, help="무작위 장면 seed")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--max_steps", type=int, default=2000)
parser.add_argument("--want", type=str, default=None,
                    help="원하는 결과(docked/entangled/finger_hit/wall_hit/timeout). "
                         "나올 때까지 seed 를 바꿔 재시도한다")
parser.add_argument("--tries", type=int, default=1, help="--want 재시도 횟수")
parser.add_argument("--linger", type=float, default=3.0,
                    help="도킹 후에도 이만큼(초) 더 기록한다. 영상이 뚝 끊기지 않게.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import math
import os

import gymnasium as gym
import numpy as np
import torch

import usvdock  # noqa: F401
from usvdock import blueboat_cfg as BB
from usvdock import geometry as G
from usvdock import perception as PC

# ---------------------------------------------------------------------------
# 고정 장면 표 — 4컷 비교와 논문 그림 재현성을 위해 명시적으로 박아 둔다.
#   berth_x, 폐어구 목록(x,y,폭,길이,상단수심), 유속, 초기(거리, 방위°, 선수오차°)
#   gear = [(dx, dy, w, l, yaw°, top), ...]  dx 는 버스 중심 기준 오프셋
# ---------------------------------------------------------------------------
SCENES = [
    # 0: 표준 — 접근 경로 정면에 폐어구 1개
    dict(berth_x=0.0, cur=+0.25, d0=9.0, bear=-18.0, yaw_err=+22.0,
         gear=[(0.2, 5.0, 0.9, 0.6, 35.0, 0.10)]),
    # 1: 무유속, 폐어구 2개 — 순수 회피 기동
    dict(berth_x=-4.0, cur=0.0, d0=8.0, bear=+12.0, yaw_err=-15.0,
         gear=[(-1.4, 4.5, 0.8, 0.55, 110.0, 0.08), (1.5, 6.0, 0.7, 0.5, 20.0, 0.15)]),
    # 2: 강한 횡류 (한계 0.479 의 94%) + 폐어구 1개
    dict(berth_x=3.0, cur=-0.45, d0=10.0, bear=+25.0, yaw_err=+30.0,
         gear=[(-0.5, 5.5, 1.0, 0.6, 70.0, 0.06)]),
    # 3: 폐어구 없음 (미션1 대조)
    dict(berth_x=1.5, cur=+0.30, d0=9.5, bear=-25.0, yaw_err=-20.0, gear=[]),
    # 5 는 아래에 추가 (발표용 설계 장면)
    # 4: 먼 거리 + 폐어구 3개 (가장 어려움)
    dict(berth_x=-2.0, cur=+0.40, d0=12.0, bear=+30.0, yaw_err=+38.0,
         gear=[(-2.0, 4.0, 0.9, 0.6, 15.0, 0.12), (0.8, 5.6, 1.0, 0.55, 130.0, 0.07),
               (2.6, 3.8, 0.7, 0.5, 60.0, 0.18)]),
    # 5: **발표용 설계 장면** — 직진 경로를 막고 우회를 강제한다.
    #    무작위 장면은 폐어구가 경로를 제대로 막지 않거나 유속이 거의 0 인 경우가 많아
    #    "회피하는 모습"이 안 보인다. 이 장면은 의도적으로 설계했다:
    #      직진 차단 O,  최소 우회폭 2.5 m,  유속 +0.30 m/s
    #    (지그재그 3개 배치는 완전 봉쇄가 되어 탈락시켰다)
    dict(berth_x=0.0, cur=+0.30, d0=9.5, bear=-5.0, yaw_err=+10.0,
         gear=[(0.0, 4.2, 0.95, 0.70, 20.0, 0.10),
               (-2.4, 6.2, 0.85, 0.65, 110.0, 0.13)]),
    # 6, 7: 장면 5 와 배치는 같고 **유속만** 낮춘 변형.
    #   유속 0.30 에서는 20개 체크포인트 전부 실패했다(최종거리 7~13 m).
    #   우회 후 복귀할 때 계속 밀리는 것이 주 원인으로 보여 유속만 분리해 확인한다.
    dict(berth_x=0.0, cur=+0.18, d0=9.5, bear=-5.0, yaw_err=+10.0,
         gear=[(0.0, 4.2, 0.95, 0.70, 20.0, 0.10),
               (-2.4, 6.2, 0.85, 0.65, 110.0, 0.13)]),
    dict(berth_x=0.0, cur=+0.08, d0=9.5, bear=-5.0, yaw_err=+10.0,
         gear=[(0.0, 4.2, 0.95, 0.70, 20.0, 0.10),
               (-2.4, 6.2, 0.85, 0.65, 110.0, 0.13)]),
    # 8, 9, 10: **제어기 건전성 시험용** — 폐어구 없음, 유속만 다름.
    #   도킹 판정은 속력 0.20 m/s 미만을 요구하는데, 유속 V 를 버티려면
    #   게걸음각으로 u·sin(δ)=V 를 만족해야 하므로 u > V 가 필요하다.
    #   즉 V ≳ 0.20 이면 두 조건이 동시에 성립할 수 없다 — 물리적으로 불가능하다.
    #   제어기가 고장난 것인지 물리 한계인지 가르려면 저유속에서 먼저 확인해야 한다.
    dict(berth_x=0.0, cur=0.00, d0=9.0, bear=-10.0, yaw_err=+15.0, gear=[]),
    dict(berth_x=0.0, cur=+0.10, d0=9.0, bear=-10.0, yaw_err=+15.0, gear=[]),
    dict(berth_x=0.0, cur=+0.20, d0=9.0, bear=-10.0, yaw_err=+15.0, gear=[]),
]


def force_scene(env, s: dict):
    """리셋 후 장면과 초기 자세를 강제 고정한다."""
    dev = env.device
    p = env._scene_p
    one = lambda v: torch.full_like(p.berth_x, float(v))  # noqa: E731
    p.berth_x[:] = one(s["berth_x"])
    p.current_u[:] = one(s["cur"])

    # 폐어구를 명시 목록으로 덮어쓴다. 지정 개수만 활성.
    # ★ 미션1(폐어구 없는 과제)에서는 장면 표의 폐어구를 무시한다.
    #   그러지 않으면 폐어구를 본 적 없는 1단계 정책에 폐어구를 던져 주게 되고,
    #   당연히 감김으로 실패한다(실제로 이 실수를 했다).
    p.gear_on[:] = False
    if bool(p.has_gear[0]):
        for i, (dx, gy, gw, gl, gyaw, gtop) in enumerate(s["gear"][: G.MAX_OBSTACLES]):
            p.gear_x[:, i] = s["berth_x"] + dx
            p.gear_y[:, i] = gy
            p.gear_w[:, i] = gw
            p.gear_l[:, i] = gl
            p.gear_yaw[:, i] = math.radians(gyaw)
            p.gear_top[:, i] = gtop
            p.gear_bot[:, i] = gtop + 2.0
            p.gear_on[:, i] = True

    env._target[:] = G.docking_target(p)

    d, bear = s["d0"], math.radians(s["bear"])
    env._eta[:, 0] = s["berth_x"] + d * math.sin(bear)
    env._eta[:, 1] = G.WALL_Y - 2.5 - d * math.cos(bear)
    env._eta[:, 2] = math.pi / 2 + math.radians(s["yaw_err"])
    env._nu[:] = 0.0
    env._thrust[:] = 0.0
    env._prev_action[:] = 0.0
    env._hold[:] = 0
    env._docked[:] = False
    env._collided[:] = False
    env._prev_dist[:] = torch.linalg.norm(env._eta[:, :2] - env._target[:, :2], dim=-1)
    env._prev_phi[:] = -env._crosstrack_potential(env._eta[:, :2])
    env._outcome_code[:] = 0
    env._dyn.reset_idx(torch.arange(env.num_envs, device=dev))
    env._write_pose()


def main():
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    cfg.scene.num_envs = 2  # 최소. 0번만 쓴다
    cfg.seed = args_cli.seed
    # 유속 커리큘럼을 끈다. 새 환경은 _env_steps=0 이라 유속이 0 으로 강제되어
    # 학습보다 쉬운 조건에서 재게 된다.
    cfg.current_warmup_steps = 0
    # 도킹으로 에피소드를 끝내지 않는다 — 이후 거동까지 기록해야 영상이 자연스럽다.
    cfg.terminate_on_dock = False
    env = gym.make(args_cli.task, cfg=cfg, render_mode=None).unwrapped

    # --- 정책 준비 ---
    if args_cli.checkpoint:
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
        wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        kind = "rl"
    else:
        from usvdock.controllers.pid import DockingPID

        pid = DockingPID(env.num_envs, env.device, env.step_dt)
        policy = None
        kind = "pid"

    # ── 원하는 결과가 나올 때까지 재시도 ─────────────────────────────────
    #   주행거리계 잡음(DVL 2 %)이 들어 있어 같은 체크포인트도 에피소드마다 다르다.
    #   발표용 4컷은 특정 거동을 보여야 하므로 seed 를 바꿔가며 찾는다.
    #   ※ 이렇게 고른 컷은 **대표 사례이지 통계가 아니다.** 수치는 eval.py 로 따로 낸다.
    saved = None
    for _try in range(max(1, args_cli.tries)):
        torch.manual_seed(args_cli.seed + _try)
        obs_dict, _ = env.reset()
        if args_cli.scene >= 0:
            force_scene(env, SCENES[args_cli.scene])
        # scene=-1 이면 환경이 뽑은 무작위 장면을 그대로 쓴다.
        # 4컷 비교용 장면을 고를 때 유용하다 — 성공하는 장면을 찾은 뒤
        # 그 파라미터를 SCENES 에 추가해 모든 체크포인트에 재현한다.

        # ★ 장면 메타데이터는 **여기서** 잡아 둔다.
        #   env.step() 은 종료 직후 내부에서 자동 리셋을 돌리며 장면을 새로 뽑는다.
        #   루프가 끝난 뒤 env._target / _scene_p 를 읽으면 **다음 에피소드의 장면**이 나온다.
        #   (도킹 성공인데 최종거리 8.8 m 로 찍혀 발견 — 그리는 버스가 딴 곳이었다)
        _p = env._scene_p
        SCENE_META = dict(
            target=env._target[0].cpu().numpy().copy(),
            berth_x=float(_p.berth_x[0]),
            gear=np.stack([_p.gear_x[0].cpu().numpy(), _p.gear_y[0].cpu().numpy(),
                           _p.gear_w[0].cpu().numpy(), _p.gear_l[0].cpu().numpy(),
                           _p.gear_yaw[0].cpu().numpy(), _p.gear_top[0].cpu().numpy()], axis=-1),
            gear_on=_p.gear_on[0].cpu().numpy(),
            current_u=float(_p.current_u[0]),
            has_gear=bool(_p.has_gear[0]),
        )
        # rsl-rl 3.x 정책은 **관측 그룹 딕셔너리**를 받는다(텐서 아님).
        obs = env._get_observations()

        s = BB.SensorMountCfg()
        rec = {k: [] for k in ("pos", "yaw", "nu", "thrust", "action", "lidar", "fls",
                               "det_x", "det_wall", "det_ok")}
        outcome, end_step, dock_step = "timeout", args_cli.max_steps, None

        for t in range(args_cli.max_steps):
            pos, yaw = env._eta[:, :2].clone(), env._eta[:, 2].clone()
            # 시각화용 스캔은 검출 해상도로 뜬다(그림이 촘촘해야 실감난다)
            # 환경 캐시를 재사용한다(스텝당 스캔 1회)
            lid, fls = env.sensors()
            # 시각화용 즉시검출 (인셋에 "지금 보이는가"를 표시)
            dx, dw, dok = PC.detect_berth(pos, yaw, env._scene_p, scan=lid)
            # 제어에 쓰는 추정은 RL 과 **동일하게** 환경의 추측항법 유지본이다
            brel, bage = env.update_berth_estimate()

            with torch.no_grad():
                if kind == "rl":
                    act = policy(obs)  # obs 는 {"policy": tensor}
                else:
                    stale = bage > 5.0  # 너무 오래된 추정이면 제자리를 목표로 둔다
                    tgt = torch.stack(
                        [torch.where(stale, pos[:, 0], pos[:, 0] + brel[:, 0]),
                         torch.where(stale, pos[:, 1],
                                     pos[:, 1] + brel[:, 1] - 0.50 - BB.LOA / 2),
                         torch.full_like(dx, math.pi / 2)], dim=-1)
                    act = pid(pos, yaw, env._nu, tgt, fls)

            rec["pos"].append(pos[0].cpu().numpy())
            rec["yaw"].append(float(yaw[0]))
            rec["nu"].append(env._nu[0].cpu().numpy())
            rec["thrust"].append(env._thrust[0].cpu().numpy())
            rec["action"].append(act[0].cpu().numpy())
            rec["lidar"].append(lid[0].cpu().numpy())
            rec["fls"].append(fls[0].cpu().numpy())
            rec["det_x"].append(float(dx[0]))
            rec["det_wall"].append(float(dw[0]))
            rec["det_ok"].append(bool(dok[0]))

            with torch.no_grad():
                obs, _, term, trunc, _ = env.step(act)

            # 도킹 시점을 기록하되 종료하지 않는다(linger 만큼 더 기록).
            if dock_step is None and bool(env._docked[0]):
                dock_step = t + 1
                outcome = "docked"
            if dock_step is not None and (t + 1 - dock_step) * env.step_dt >= args_cli.linger:
                end_step = t + 1
                break

            if bool(term[0]) or bool(trunc[0]):
                end_step = t + 1
                # ★ 이미 도킹에 성공한 뒤라면 결과를 덮어쓰지 않는다.
                #   linger(도킹 후 추가 기록) 중에 배가 밀려 핑거에 닿으면
                #   성공한 에피소드가 finger_hit 으로 잘못 기록된다.
                #   도킹 후 접촉은 실제 마리나에서도 방충재가 받아주는 정상 상황이다.
                if dock_step is None:
                    # step() 내부 자동 리셋이 _docked/_entangled 를 지우므로
                    # 리셋에 살아남는 _outcome_code 를 읽는다.
                    outcome = {0: "timeout", 1: "docked", 2: "entangled",
                               3: "finger_hit", 4: "wall_hit"}[int(env._outcome_code[0])]
                break

        out = {k: np.asarray(v) for k, v in rec.items()}
        out.update(SCENE_META)
        out.update(
            outcome=outcome, end_step=end_step, dt=env.step_dt,
            dock_step=(-1 if dock_step is None else dock_step),
            checkpoint=args_cli.checkpoint or "PID", scene=args_cli.scene, task=args_cli.task,
        )

        d = float(np.linalg.norm(out["pos"][-1] - out["target"][:2]))
        spd = float(np.linalg.norm(out["nu"][-1][:2]))
        if args_cli.scene < 0:
            g = SCENE_META
            on = _p.gear_on[0].cpu().numpy()
            gl = [(round(float(_p.gear_x[0, i] - _p.berth_x[0]), 2), round(float(_p.gear_y[0, i]), 2),
                   round(float(_p.gear_w[0, i]), 2), round(float(_p.gear_l[0, i]), 2),
                   round(math.degrees(float(_p.gear_yaw[0, i])), 1), round(float(_p.gear_top[0, i]), 3))
                  for i in range(len(on)) if on[i]]
            print(f"[SCENE] berth_x={g['berth_x']:.2f} cur={g['current_u']:+.2f} gear={gl}", flush=True)
        print(f"[ROLL] {os.path.basename(args_cli.out)}  결과={outcome}  "
              f"{end_step} 스텝({end_step*env.step_dt:.1f}s)  "
              f"최종거리={d:.3f}m  최종속력={spd:.3f}m/s", flush=True)
        if saved is None or (args_cli.want and outcome == args_cli.want):
            saved = (out, outcome, end_step)
        if args_cli.want is None or outcome == args_cli.want:
            break
        print(f"[TRY {_try}] 결과={outcome} (원하는 것: {args_cli.want}) 재시도", flush=True)
    out, outcome, end_step = saved
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    np.savez_compressed(args_cli.out, **out)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
