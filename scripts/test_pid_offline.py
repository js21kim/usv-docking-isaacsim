"""PID + 3자유도 동역학 오프라인 궤적 시험 (Isaac Sim 불필요).

Isaac Lab 안에서 PID 가 0% 성공·100% 시간초과였다. 도킹은커녕 접근을 못 한다는 뜻이라
보상 문제가 아니라 제어/동역학/부호 규약 중 하나가 깨진 것이다.
여기서 순수 torch 로 재현해 어디가 문제인지 분리한다.
"""

import math
import sys
import types
import importlib.util

import torch


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


pkg = types.ModuleType("usvdock")
pkg.__path__ = ["/home/jason/07_USV_Docking_IsaacSim/usvdock"]
sys.modules["usvdock"] = pkg
BASE = "/home/jason/07_USV_Docking_IsaacSim/usvdock"
C = _load("usvdock.blueboat_cfg", f"{BASE}/blueboat_cfg.py")
G = _load("usvdock.geometry", f"{BASE}/geometry.py")
D = _load("usvdock.dynamics", f"{BASE}/dynamics.py")
ctrl_pkg = types.ModuleType("usvdock.controllers")
ctrl_pkg.__path__ = [f"{BASE}/controllers"]
sys.modules["usvdock.controllers"] = ctrl_pkg
PID = _load("usvdock.controllers.pid", f"{BASE}/controllers/pid.py")


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def run(n=4, steps=2000, dt=0.02, current=0.0, verbose=True):
    dev = "cpu"
    torch.manual_seed(3)
    p = G.sample_scene(n, dev, mission=1, current_ms=current)
    p.berth_x[:] = 0.0

    dyn = D.SurfaceDynamics3DOF(n, dev, dt=dt)
    pid = PID.DockingPID(n, dev, dt)

    # 정면 8 m, 선수 정렬 — 가장 쉬운 조건부터
    eta = torch.zeros(n, 3)
    eta[:, 0] = 0.0
    eta[:, 1] = G.WALL_Y - 2.5 - 8.0
    eta[:, 2] = math.pi / 2
    nu = torch.zeros(n, 3)

    target = G.docking_target(p)
    thrust = torch.zeros(n, 2)
    tau_lag = 0.15

    hist = []
    for k in range(steps):
        act = pid(eta[:, :2], eta[:, 2], nu, target)
        cmd = act * C.THRUST_MAX_PER_MOTOR
        a = dt / (tau_lag + dt)
        thrust = thrust + a * (cmd - thrust)

        c, s = torch.cos(eta[:, 2]), torch.sin(eta[:, 2])
        nu_c = torch.stack([p.current_u * c, -p.current_u * s, torch.zeros(n)], dim=-1)
        tau = dyn.allocate(thrust[:, 0], thrust[:, 1])
        f = dyn.fluid_wrench(nu, nu_c)
        m_eff = dyn.m_rb + dyn.m_a
        nu = (nu + dt * (tau + f) / m_eff).clamp(-5, 5)

        c, s = torch.cos(eta[:, 2]), torch.sin(eta[:, 2])
        eta[:, 0] += dt * (nu[:, 0] * c - nu[:, 1] * s)
        eta[:, 1] += dt * (nu[:, 0] * s + nu[:, 1] * c)
        eta[:, 2] = wrap(eta[:, 2] + dt * nu[:, 2])

        if k % 200 == 0 or k == steps - 1:
            d = torch.linalg.norm(eta[:, :2] - target[:, :2], dim=-1)
            hist.append((k * dt, float(eta[0, 0]), float(eta[0, 1]), math.degrees(float(eta[0, 2])),
                         float(nu[0, 0]), float(d[0]), float(act[0, 0]), float(act[0, 1])))

    if verbose:
        print(f"목표: x={float(target[0,0]):.2f} y={float(target[0,1]):.2f} "
              f"yaw={math.degrees(float(target[0,2])):.0f}°   유속 {current} m/s")
        print(f"{'t[s]':>6} {'x':>7} {'y':>7} {'yaw°':>7} {'u[m/s]':>8} {'거리':>7} "
              f"{'act_p':>7} {'act_s':>7}")
        for h in hist:
            print(f"{h[0]:6.1f} {h[1]:7.2f} {h[2]:7.2f} {h[3]:7.1f} {h[4]:8.3f} {h[5]:7.2f} "
                  f"{h[6]:7.3f} {h[7]:7.3f}")
    d_fin = float(torch.linalg.norm(eta[0, :2] - target[0, :2]))
    return d_fin


if __name__ == "__main__":
    print("=" * 78)
    print("무유속, 정면 8 m, 선수 정렬 — 가장 쉬운 조건")
    print("=" * 78)
    d = run(current=0.0)
    print(f"\n최종 거리 {d:.3f} m  →  {'도달' if d < 0.4 else '실패'}")
