"""고정 장면 하나에서 체크포인트를 훑어 거동을 분류한다.

발표용 4컷을 고르려면 "감김 / 핑거충돌 / 성공" 세 거동을 같은 장면에서 보이는
체크포인트를 찾아야 한다. 롤아웃을 매번 띄우면 40초씩 낭비되므로
Isaac Sim 을 한 번만 띄우고 순회한다.
"""
import argparse
from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--run_dir", required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--scene", type=int, default=5)
ap.add_argument("--stride", type=int, default=150)
ap.add_argument("--min_iter", type=int, default=0)
ap.add_argument("--max_iter", type=int, default=10**9)
AppLauncher.add_app_launcher_args(ap)
a = ap.parse_args()
sim = AppLauncher(a).app

import glob, os, re, torch, gymnasium as gym
import usvdock  # noqa
import importlib.util as _il
_sp = _il.spec_from_file_location("_ro", os.path.join(os.path.dirname(__file__), "rollout.py"))

# rollout.py 의 SCENES/force_scene 을 재사용하기 위해 최소한만 복제
from usvdock import geometry as G
import math

SCENES = None
with open(os.path.join(os.path.dirname(__file__), "rollout.py")) as f:
    src = f.read()
ns = {}
exec(src[src.index("SCENES = ["): src.index("def force_scene")], {"math": math}, ns)
SCENES = ns["SCENES"]


def force_scene(env, s):
    p = env._scene_p
    one = lambda v: torch.full_like(p.berth_x, float(v))
    p.berth_x[:] = one(s["berth_x"]); p.current_u[:] = one(s["cur"])
    p.gear_on[:] = False
    if bool(p.has_gear[0]):
        for i, (dx, gy, gw, gl, gyaw, gtop) in enumerate(s["gear"][: G.MAX_OBSTACLES]):
            p.gear_x[:, i] = s["berth_x"] + dx; p.gear_y[:, i] = gy
            p.gear_w[:, i] = gw; p.gear_l[:, i] = gl
            p.gear_yaw[:, i] = math.radians(gyaw); p.gear_top[:, i] = gtop
            p.gear_bot[:, i] = gtop + 2.0; p.gear_on[:, i] = True
    env._target[:] = G.docking_target(p)
    d, bear = s["d0"], math.radians(s["bear"])
    env._eta[:, 0] = s["berth_x"] + d * math.sin(bear)
    env._eta[:, 1] = G.WALL_Y - 2.5 - d * math.cos(bear)
    env._eta[:, 2] = math.pi / 2 + math.radians(s["yaw_err"])
    env._nu[:] = 0.0; env._thrust[:] = 0.0; env._prev_action[:] = 0.0
    env._hold[:] = 0; env._docked[:] = False; env._collided[:] = False
    env._outcome_code[:] = 0
    env._prev_dist[:] = torch.linalg.norm(env._eta[:, :2] - env._target[:, :2], dim=-1)
    env._prev_phi[:] = -env._crosstrack_potential(env._eta[:, :2])
    env._dyn.reset_idx(torch.arange(env.num_envs, device=env.device))
    env._write_pose()


cfg = gym.spec(a.task).kwargs["env_cfg_entry_point"]
cfg.scene.num_envs = 2; cfg.seed = 1; cfg.current_warmup_steps = 0
env = gym.make(a.task, cfg=cfg, render_mode=None).unwrapped

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner
agent_cfg = gym.spec(a.task).kwargs["rsl_rl_cfg_entry_point"]
runner = OnPolicyRunner(RslRlVecEnvWrapper(env), agent_cfg.to_dict(), log_dir=None, device=env.device)

NAME = {0: "timeout", 1: "DOCKED", 2: "entangled", 3: "finger_hit", 4: "wall_hit"}
ckpts = sorted(glob.glob(os.path.join(a.run_dir, "model_*.pt")),
               key=lambda f: int(re.search(r"model_(\d+)", f).group(1)))
picked = [c for c in ckpts
          if int(re.search(r"model_(\d+)", c).group(1)) % a.stride == 0
          and a.min_iter <= int(re.search(r"model_(\d+)", c).group(1)) <= a.max_iter]
print(f"[SWEEP] scene {a.scene}, 체크포인트 {len(picked)}개", flush=True)

for c in picked:
    it = int(re.search(r"model_(\d+)", c).group(1))
    runner.load(c); pol = runner.get_inference_policy(device=env.device)
    obs, _ = env.reset(); force_scene(env, SCENES[a.scene]); obs = env._get_observations()
    out, n = "timeout", int(env.max_episode_length)
    # ★ 거리·이격은 step() **전에** 읽는다. step() 안의 자동 리셋이 상태를 갈아치운다.
    #   (이전에는 리셋 후 값을 읽어 DOCKED 인데 최종거리 11.28 m 로 찍혔다)
    d = float("nan"); clear = 1e3; ymax = -1e3
    for t in range(n):
        with torch.no_grad():
            d = float(torch.linalg.norm(env._eta[0, :2] - env._target[0, :2]))
            col = G.check_collision(env._eta[:, :2], env._eta[:, 2], env._scene_p)
            clear = min(clear, float(col["clearance"][0]))
            ymax = max(ymax, float(env._eta[0, 1]))
            obs, _, term, trunc, _ = env.step(pol(obs))
        if bool(term[0]) or bool(trunc[0]):
            out = NAME[int(env._outcome_code[0])]; n = t + 1; break
    # 폐어구를 지났는가: 가장 깊이 들어간 y 가 가장 먼 폐어구보다 벽 쪽인가
    gy = float(env._scene_p.gear_y[0][env._scene_p.gear_on[0]].max())
    passed = "O" if ymax > gy else "X"
    cs = f"{clear:5.2f}" if clear < 1e2 else "  -  "
    print(f"  iter {it:5d}  {out:11} {n*env.step_dt:5.1f}s  "
          f"목표거리 {d:5.2f} m  최소이격 {cs} m  폐어구통과 {passed}", flush=True)
env.close()
sim.close()
