"""2단계 비교 평가 — 각 단계에서 PID vs RL.

    1단계 (쉬운 환경, 폐어구 없음)
        PID(LiDAR)  vs  RL(LiDAR)
        주장: 부족구동 + 횡류에서 RL 이 낫다

    2단계 (어려운 환경, 경로상 폐어구)
        PID(LiDAR+FLS)  vs  RL(LiDAR+FLS)
        주장: 같은 접근이 위험 환경으로 확장된다
        + 절제(--checkpoint_ablation): RL(LiDAR only) → 폐어구를 못 보고 돌진

공정성 장치:
  - 같은 seed 로 장면을 샘플링하므로 모든 방식이 **완전히 동일한 에피소드**를 푼다
  - PID 도 RL 과 **같은 검출기**(perception.detect_berth)를 쓴다. 정답 좌표를 쓰지 않는다
  - PID 의 회피도 **현재 스캔에 즉시 반응**하는 방식이다. 전역 경로계획(A*, RRT)을 주면
    PID 쪽에만 지도와 계획이 생겨 비교가 교란된다

실행:
    python scripts/eval.py --stage 1 --episodes 512 --checkpoint_rl <B_M1 모델>
    python scripts/eval.py --stage 2 --episodes 512 --checkpoint_rl <C_M2 모델> \
        --checkpoint_ablation <B_M2 모델>
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="2단계 PID vs RL 비교")
parser.add_argument("--stage", type=int, required=True, choices=(1, 2))
parser.add_argument("--episodes", type=int, default=512)
parser.add_argument("--checkpoint_rl", type=str, default=None, help="해당 단계의 RL 정책")
parser.add_argument("--checkpoint_ablation", type=str, default=None,
                    help="2단계 전용: FLS 없는 RL 정책")
parser.add_argument("--current", type=str, default="0,0.45",
                    help="유속 범위 lo,hi [m/s]. 정횡 한계 0.479 를 넘기지 말 것")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import math
import os

import gymnasium as gym
import torch

import usvdock  # noqa: F401
from usvdock import blueboat_cfg as BB
from usvdock import geometry as G
from usvdock import perception as PC
from usvdock.controllers.pid import DockingPID


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def make_env(mission: int, use_fls: bool, n: int, cur, seed: int):
    task = f"Isaac-USVDock-M{mission}-{'Fusion' if use_fls else 'Lidar'}-v0"
    cfg = gym.spec(task).kwargs["env_cfg_entry_point"]
    cfg.scene.num_envs = n
    cfg.current_range = cur
    cfg.seed = seed
    return gym.make(task, cfg=cfg, render_mode=None).unwrapped, task


def run(env, policy, n_steps):
    """정책을 돌려 에피소드 결과를 집계한다.

    ※ env.step() 은 종료 직후 내부에서 자동 리셋을 돌리며 _docked/_entangled 를 지운다.
      따라서 결과는 리셋에 살아남는 _outcome_code 에서 읽는다.
      (0=시간초과 1=도킹 2=폐어구감김 3=핑거 4=벽)
    """
    obs, _ = env.reset()
    N, dev = env.num_envs, env.device
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    code = torch.zeros(N, dtype=torch.long, device=dev)
    t_end = torch.full((N,), float("nan"), device=dev)
    pos_err = torch.full((N,), float("nan"), device=dev)
    yaw_err = torch.full((N,), float("nan"), device=dev)

    for step in range(n_steps):
        with torch.inference_mode():
            act = policy(obs, env)
            # 오차는 step() **전에** 읽는다. step() 안의 자동 리셋이 상태를 바꾼다.
            d = torch.linalg.norm(env._eta[:, :2] - env._target[:, :2], dim=-1)
            y = _wrap(env._eta[:, 2] - env._target[:, 2]).abs()
            obs, _, term, trunc, _ = env.step(act)

        newly = (term | trunc) & (~done_once)
        if newly.any():
            code = torch.where(newly, env._outcome_code, code)
            t_end = torch.where(newly, torch.full_like(t_end, step * env.step_dt), t_end)
            pos_err = torch.where(newly, d, pos_err)
            yaw_err = torch.where(newly, y, yaw_err)
            done_once |= newly
        if done_once.all():
            break

    f = lambda m: float(m.float().mean())  # noqa: E731
    ok = code == 1
    return {
        "success_rate": f(ok),
        "entangle_rate": f(code == 2),
        "finger_hit_rate": f(code == 3),
        "wall_hit_rate": f(code == 4),
        "timeout_rate": f(code == 0),
        "pos_err_mean": float(pos_err[ok].mean()) if ok.any() else float("nan"),
        "yaw_err_deg_mean": float(torch.rad2deg(yaw_err[ok]).mean()) if ok.any() else float("nan"),
        "time_to_dock_s": float(t_end[ok].mean()) if ok.any() else float("nan"),
        "n_episodes": int(env.num_envs),
    }


def make_pid_policy(env, use_fls: bool):
    """PID 정책. RL 과 같은 검출기를 쓰고, 2단계에서는 FLS 회피를 켠다.

    detect_berth 는 (상대 x, 벽까지 거리, 유효)를 준다. 전부 스캔에서 나온 값이며
    참 위치를 쓰지 않는다 — RL 과 정보 조건이 같아야 비교가 성립한다.
    """
    pid = DockingPID(env.num_envs, env.device, env.step_dt)

    def policy(obs, e):
        pos, yaw = e._eta[:, :2], e._eta[:, 2]
        rel_x, wall_d, ok = PC.detect_berth(pos, yaw, e._scene_p)
        # 검출 실패 시 제자리를 목표로 두어 표류를 막는다
        tx = torch.where(ok, pos[:, 0] + rel_x, pos[:, 0])
        ty = torch.where(ok, pos[:, 1] + wall_d - 0.50 - BB.LOA / 2, pos[:, 1])
        tgt = torch.stack([tx, ty, torch.full_like(tx, math.pi / 2)], dim=-1)
        fls = G.fls_scan(pos, yaw, e._scene_p) if use_fls else None
        return pid(pos, yaw, e._nu, tgt, fls)

    return policy


def load_rl(env, task, ckpt):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = gym.spec(task).kwargs["rsl_rl_cfg_entry_point"]
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(ckpt)
    inf = runner.get_inference_policy(device=env.device)
    return lambda o, e: inf(o)


def main():
    lo, hi = (float(v) for v in args_cli.current.split(","))
    cur, N = (lo, hi), args_cli.episodes
    stage = args_cli.stage
    mission = 1 if stage == 1 else 2
    use_fls = stage == 2
    results, order = {}, []

    def add(name, env, policy):
        print(f"[EVAL] {name}", flush=True)
        results[name] = run(env, policy, int(env.max_episode_length))
        order.append(name)
        env.close()

    sensors = "LiDAR+FLS" if use_fls else "LiDAR"

    # --- PID ---
    env, task = make_env(mission, use_fls, N, cur, args_cli.seed)
    add(f"PID ({sensors})", env, make_pid_policy(env, use_fls))

    # --- RL ---
    if args_cli.checkpoint_rl and os.path.isfile(args_cli.checkpoint_rl):
        env, task = make_env(mission, use_fls, N, cur, args_cli.seed)
        add(f"RL ({sensors})", env, load_rl(env, task, args_cli.checkpoint_rl))
    else:
        print(f"[EVAL] RL 건너뜀 (체크포인트 없음: {args_cli.checkpoint_rl})", flush=True)

    # --- 절제 (2단계 전용) ---
    if stage == 2 and args_cli.checkpoint_ablation and os.path.isfile(args_cli.checkpoint_ablation):
        env, task = make_env(2, False, N, cur, args_cli.seed)
        add("RL (LiDAR only) — ablation", env, load_rl(env, task, args_cli.checkpoint_ablation))

    # --- 출력 ---
    title = ("Stage 1 — easy (no derelict gear)" if stage == 1
             else "Stage 2 — hard (derelict gear on approach path)")
    print()
    print("=" * 96)
    print(f"{title} | current {lo}~{hi} m/s | {N} episodes | seed {args_cli.seed}")
    print("=" * 96)
    print(f"{'method':30} {'success':>8} {'entangle':>9} {'finger':>8} {'wall':>7} "
          f"{'timeout':>8} {'pos err':>9} {'yaw err':>8} {'t_dock':>8}")
    print("-" * 96)
    for k in order:
        r = results[k]
        print(f"{k:30} {100*r['success_rate']:7.1f}% {100*r['entangle_rate']:8.1f}% "
              f"{100*r['finger_hit_rate']:7.1f}% {100*r['wall_hit_rate']:6.1f}% "
              f"{100*r['timeout_rate']:7.1f}% {r['pos_err_mean']:8.3f}m "
              f"{r['yaw_err_deg_mean']:7.1f}° {r['time_to_dock_s']:7.1f}s")
    print("=" * 96)

    out = args_cli.out or f"results/eval_stage{stage}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump({"stage": stage, "mission": mission, "current": cur,
                   "seed": args_cli.seed, "results": results}, fp, indent=2, ensure_ascii=False)
    print(f"[EVAL] 저장: {out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
