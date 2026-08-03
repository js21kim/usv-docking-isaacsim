"""gym 등록.

marinelab / constrained-albc 과 같은 구조다 — 패키지를 import 해야 등록이 일어난다.
isaaclab_tasks 는 외부 패키지의 entry-point 를 소비하지 않으므로
(06_ALBC_IsaacSim 작업기록 참조) 스크립트에서 `import usvdock` 이 필요하다.

Task ID:
    Isaac-USVDock-M1-Lidar-v0    미션1, Arm B (LiDAR)
    Isaac-USVDock-M1-Fusion-v0   미션1, Arm C (LiDAR+FLS)  ← 통제군: B와 같아야 정상
    Isaac-USVDock-M2-Lidar-v0    미션2, Arm B
    Isaac-USVDock-M2-Fusion-v0   미션2, Arm C  ← 여기서 차이가 나야 한다
"""

import gymnasium as gym

from .agents import DockingPPORunnerCfg
from .docking_env import DockingEnv
from .docking_env_cfg import DockingEnvCfg

__all__ = ["DockingEnv", "DockingEnvCfg", "DockingPPORunnerCfg"]


def _cfg(mission: int, use_fls: bool) -> DockingEnvCfg:
    c = DockingEnvCfg()
    c.mission = mission
    c.use_fls = use_fls
    return c


for _m in (1, 2):
    for _name, _fls in (("Lidar", False), ("Fusion", True)):
        gym.register(
            id=f"Isaac-USVDock-M{_m}-{_name}-v0",
            entry_point="usvdock.envs.docking_env:DockingEnv",
            disable_env_checker=True,
            kwargs={
                "env_cfg_entry_point": _cfg(_m, _fls),
                "rsl_rl_cfg_entry_point": DockingPPORunnerCfg(),
            },
        )
