"""스텝당 소요 시간 분해 — 어디가 느린지 직접 잰다. (일회성 진단 도구)"""
import argparse
from isaaclab.app import AppLauncher
ap = argparse.ArgumentParser(); ap.add_argument("--stage", type=int, default=1)
ap.add_argument("--episodes", type=int, default=128); ap.add_argument("--steps", type=int, default=200)
AppLauncher.add_app_launcher_args(ap); a = ap.parse_args()
sim = AppLauncher(a).app
import time, math, torch, gymnasium as gym
import usvdock  # noqa
from usvdock import perception as PC, blueboat_cfg as BB
from usvdock.controllers.pid import DockingPID

mission, use_fls = (1, False) if a.stage == 1 else (2, True)
task = f"Isaac-USVDock-M{mission}-{'Fusion' if use_fls else 'Lidar'}-v0"
cfg = gym.spec(task).kwargs["env_cfg_entry_point"]
cfg.scene.num_envs = a.episodes; cfg.current_range = (0.0, 0.45)
cfg.seed = 1; cfg.current_warmup_steps = 0
env = gym.make(task, cfg=cfg, render_mode=None).unwrapped
pid = DockingPID(env.num_envs, env.device, env.step_dt)
obs, _ = env.reset()
def sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()
T = {k: 0.0 for k in ("sensors","detect","pid","step")}
t_all = time.time()
for i in range(a.steps):
    t0=time.time(); pos,yaw = env._eta[:,:2], env._eta[:,2]
    lid,fls = env.sensors(); sync(); t1=time.time()
    rx,wd,ok = PC.detect_berth(pos,yaw,env._scene_p,scan=lid); sync(); t2=time.time()
    tx=torch.where(ok,pos[:,0]+rx,pos[:,0]); ty=torch.where(ok,pos[:,1]+wd-0.50-BB.LOA/2,pos[:,1])
    tgt=torch.stack([tx,ty,torch.full_like(tx,math.pi/2)],dim=-1)
    act=pid(pos,yaw,env._nu,tgt, fls if use_fls else None); sync(); t3=time.time()
    obs,_,term,trunc,_ = env.step(act); sync(); t4=time.time()
    T["sensors"]+=t1-t0; T["detect"]+=t2-t1; T["pid"]+=t3-t2; T["step"]+=t4-t3
tot=time.time()-t_all; n=a.steps
print(f"\n=== stage{a.stage}, {a.episodes} env, {n} step ===")
for k in ("sensors","detect","pid","step"):
    print(f"  {k:9} {1000*T[k]/n:8.2f} ms/step  ({100*T[k]/tot:5.1f}%)")
print(f"  {'total':9} {1000*tot/n:8.2f} ms/step   → 2000 스텝 {tot/n*2000/60:.1f} 분")
env.close()
