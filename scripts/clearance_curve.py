"""체크포인트별 이격거리·성능 곡선 — "학습될수록 아슬아슬해진다"의 근거.

학습 로그에서 이격거리를 뽑지 않는 이유:
  rsl-rl 은 매 스텝의 log 값을 평균낸다. "성공 에피소드의 최소 이격거리"를 내보내면
  성공 에피소드가 없는 스텝이 0 을 기여해 값이 희석된다(실제로 path_len 이 0.65 m 로
  찍혔다 — 6~12 m 를 이동하는데 불가능한 값). 정확한 수치는 여기서 측정한다.

측정 내용 (체크포인트마다 N 에피소드):
  success / entangle / finger / wall / timeout 비율
  성공 에피소드의 **최소 이격거리** 분포 (중앙값, 5·95 분위)
  성공 에피소드의 경로 길이
  → 규정 안전여유(0.5 m)로 **위에서 수렴**하는 곡선이 나와야 한다

Isaac Sim 을 한 번만 띄우고 체크포인트를 순회한다(매번 띄우면 40초씩 낭비).

실행:
    python scripts/clearance_curve.py --run_dir logs/rsl_rl/usv_docking/C_M2_xxx \
        --task Isaac-USVDock-M2-Fusion-v0 --episodes 256 --stride 100
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="체크포인트별 이격거리 곡선")
parser.add_argument("--run_dir", type=str, required=True)
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--episodes", type=int, default=256)
parser.add_argument("--stride", type=int, default=100, help="체크포인트 간격")
parser.add_argument("--current", type=str, default="0,0.45")
parser.add_argument("--seed", type=int, default=4242)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import glob
import json
import os
import re

import gymnasium as gym
import torch

import usvdock  # noqa: F401
from usvdock import geometry as G


def measure(env, policy, n_steps: int) -> dict:
    """한 정책을 굴려 성능과 이격거리를 측정한다."""
    obs, _ = env.reset()
    N, dev = env.num_envs, env.device
    done_once = torch.zeros(N, dtype=torch.bool, device=dev)
    code = torch.zeros(N, dtype=torch.long, device=dev)
    min_clear = torch.full((N,), 1e3, device=dev)
    path = torch.zeros(N, device=dev)

    for _ in range(n_steps):
        with torch.no_grad():
            act = policy(obs)
            # 이격거리와 경로는 step() **전에** 누적한다(자동 리셋이 상태를 바꾼다)
            col = G.check_collision(env._eta[:, :2], env._eta[:, 2], env._scene_p)
            live = ~done_once
            min_clear = torch.where(live, torch.minimum(min_clear, col["clearance"]), min_clear)
            path = torch.where(
                live, path + torch.linalg.norm(env._nu[:, :2], dim=-1) * env.step_dt, path
            )
            obs, _, term, trunc, _ = env.step(act)

        newly = (term | trunc) & (~done_once)
        if newly.any():
            code = torch.where(newly, env._outcome_code, code)
            done_once |= newly
        if done_once.all():
            break

    ok = code == 1
    f = lambda m: float(m.float().mean())  # noqa: E731
    res = {
        "success_rate": f(ok),
        "entangle_rate": f(code == 2),
        "finger_hit_rate": f(code == 3),
        "wall_hit_rate": f(code == 4),
        "timeout_rate": f(code == 0),
        "n_success": int(ok.sum()),
    }
    if ok.any():
        c = min_clear[ok]
        c = c[torch.isfinite(c) & (c < 1e2)]  # 폐어구가 멀거나 없는 경우 제외
        if c.numel() > 0:
            res.update(
                clear_median=float(c.median()),
                clear_p05=float(c.quantile(0.05)),
                clear_p95=float(c.quantile(0.95)),
                clear_mean=float(c.mean()),
            )
        res["path_len_median"] = float(path[ok].median())
    return res


def main():
    lo, hi = (float(v) for v in args_cli.current.split(","))
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    cfg.scene.num_envs = args_cli.episodes
    cfg.current_range = (lo, hi)
    cfg.seed = args_cli.seed
    # 유속 커리큘럼을 끈다. 새 환경은 _env_steps=0 이라 유속이 0 으로 강제되어
    # 학습보다 쉬운 조건에서 재게 된다.
    cfg.current_warmup_steps = 0
    env = gym.make(args_cli.task, cfg=cfg, render_mode=None).unwrapped

    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner

    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    wrapped = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=env.device)

    ckpts = sorted(
        glob.glob(os.path.join(args_cli.run_dir, "model_*.pt")),
        key=lambda f: int(re.search(r"model_(\d+)\.pt", f).group(1)),
    )
    picked = [c for c in ckpts
              if int(re.search(r"model_(\d+)\.pt", c).group(1)) % args_cli.stride == 0]
    if ckpts and ckpts[-1] not in picked:
        picked.append(ckpts[-1])
    print(f"[CURVE] 체크포인트 {len(picked)}개 (전체 {len(ckpts)})", flush=True)

    rows = []
    steps = int(env.max_episode_length)
    for c in picked:
        it = int(re.search(r"model_(\d+)\.pt", c).group(1))
        runner.load(c)
        pol = runner.get_inference_policy(device=env.device)
        r = measure(env, pol, steps)
        r["iteration"] = it
        rows.append(r)
        print(
            f"  iter {it:5d}  성공 {100*r['success_rate']:5.1f}%  "
            f"감김 {100*r['entangle_rate']:5.1f}%  "
            f"이격중앙 {r.get('clear_median', float('nan')):6.3f} m  "
            f"경로 {r.get('path_len_median', float('nan')):6.2f} m",
            flush=True,
        )

    out = args_cli.out or f"results/clearance_{os.path.basename(args_cli.run_dir)}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fp:
        json.dump({"task": args_cli.task, "run_dir": args_cli.run_dir,
                   "current": [lo, hi], "episodes": args_cli.episodes,
                   "safety_margin": 0.50, "rows": rows}, fp, indent=2)
    print(f"[CURVE] 저장: {out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
