"""usvdock — BlueBoat 수상 자율 도킹 (Isaac Sim / Isaac Lab).

`import usvdock` 이 task 등록을 촉발한다. 이는 marinelab / constrained-albc 과 같은
구조이며, isaaclab_tasks 가 외부 패키지의 entry-point 를 소비하지 않기 때문이다.
"""

from . import envs  # noqa: F401  gym.register() 촉발

__version__ = "0.1.0"
