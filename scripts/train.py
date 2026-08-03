"""도킹 정책 학습 (rsl-rl PPO).

실행 (컨테이너 안):
    cd /workspace/usvdock
    python scripts/train.py --task Isaac-USVDock-M2-Fusion-v0 --num_envs 2048 \
        --max_iterations 600 --headless

※ PYTHONUNBUFFERED=1 필수. Isaac Sim 의 simulation_app.close() 가 stdout 버퍼를
  비우지 않고 종료해 print 출력이 통째로 사라진다 (06_ALBC_IsaacSim 에서 실측).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="USV 도킹 PPO 학습")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--run_name", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
from datetime import datetime

import gymnasium as gym
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import usvdock  # noqa: F401  gym 등록 촉발


def main():
    env_cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]
    agent_cfg = gym.spec(args_cli.task).kwargs["rsl_rl_cfg_entry_point"]
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.max_iterations:
        agent_cfg.max_iterations = args_cli.max_iterations

    # 로그 경로는 **cwd 상대**다. /workspace/usvdock 에서 실행하면 bind mount 안으로
    # 떨어져 호스트에서 바로 보인다 (06_ALBC_IsaacSim RUN.md 규칙 2).
    stamp = datetime.now().strftime("%y%m%d_%H%M%S")
    tag = args_cli.run_name or args_cli.task.replace("Isaac-USVDock-", "").replace("-v0", "")
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name,
                                           f"{tag}_{stamp}"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] 로그: {log_dir}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()
    print(f"[INFO] 학습 완료: {log_dir}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
