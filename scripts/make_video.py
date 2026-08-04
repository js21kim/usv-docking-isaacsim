"""학습 진행 4컷 영상 — 같은 장면을 서로 다른 학습 시점의 정책으로 푼다.

각 패널에 담는 것 (사용자 요청):
  - top-down 수조 전경, 버스(핑거 2개), 수중 기초부
  - BlueBoat 쌍동선 형상 + **전방 화살표**
  - LiDAR 시야와 반사점 / FLS 부채꼴과 반사점
  - 취득 데이터: 검출된 버스 중심, 거리·속력·추력 게이지
  - 지나온 항적

실행:
    python scripts/make_video.py traj/a.npz traj/b.npz traj/c.npz traj/d.npz \\
        --out figures/learning_progress.mp4
"""

import argparse
import math
import os
import sys
import importlib.util
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

ROOT = "/home/jason/07_USV_Docking_IsaacSim"


def _load(n, p):
    sp = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(sp)
    sys.modules[n] = m
    sp.loader.exec_module(m)
    return m


pkg = types.ModuleType("usvdock")
pkg.__path__ = [f"{ROOT}/usvdock"]
sys.modules["usvdock"] = pkg
C = _load("usvdock.blueboat_cfg", f"{ROOT}/usvdock/blueboat_cfg.py")
G = _load("usvdock.geometry", f"{ROOT}/usvdock/geometry.py")

CB = dict(hull="#E8833A", hull_e="#7E5109", deck="#F5CBA7", lidar="#2E86C1",
          fls="#27AE60", wall="#5D6D7E", finger="#B7950B", footing="#922B21",
          water="#D6EAF8", trail="#34495E", det="#C0392B")
OUTCOME_KO = {"docked": "DOCKED", "entangled": "NET ENTANGLEMENT",
              "finger_hit": "FINGER COLLISION", "wall_hit": "WALL COLLISION",
              "timeout": "TIMEOUT"}
OUTCOME_COL = {"docked": "#1E8449", "entangled": "#922B21",
               "finger_hit": "#B03A2E", "wall_hit": "#B03A2E", "timeout": "#7D6608"}


class Panel:
    """한 패널 = 한 체크포인트의 궤적."""

    def __init__(self, ax, d, title):
        self.ax, self.d = ax, d
        self.n = int(d["end_step"])
        self.dt = float(d["dt"])
        self.outcome = str(d["outcome"])

        bx = float(d["berth_x"])
        tgt = d["target"]

        # --- 정적 배경 ---
        ax.set_facecolor(CB["water"])
        ax.set_aspect("equal")
        ax.set_xlim(bx - 8.5, bx + 8.5)
        ax.set_ylim(G.WALL_Y - 15.0, G.WALL_Y + 1.0)
        ax.set_xticks([]); ax.set_yticks([])

        # 벽
        ax.add_patch(mp.Rectangle((bx - 20, G.WALL_Y), 40, 3, fc=CB["wall"], zorder=3))
        # 폐어구 — LiDAR 에 원리적으로 안 보이는 것.
        # 그물 느낌을 주려고 격자 해칭을 쓴다. 수면 아래라는 것을 색으로 구분한다.
        gear, gon = d["gear"], d["gear_on"]
        margin = C.BerthCfg().gear_safety_margin
        for i in range(len(gon)):
            if not bool(gon[i]):
                continue
            gx, gy, gw, gl, gyaw, gtop = [float(v) for v in gear[i]]
            deg = math.degrees(gyaw)
            # 안전 이격 영역 — 판정 경계. 정책은 이 경계를 따라 최단으로 지나가야 한다.
            ax.add_patch(mp.Rectangle((gx - gw / 2 - margin, gy - gl / 2 - margin),
                                      gw + 2 * margin, gl + 2 * margin,
                                      angle=deg, rotation_point=(gx, gy),
                                      fc="none", ec=CB["footing"], ls="--", lw=1.0,
                                      alpha=0.55, zorder=1))
            # 그물 실체 (yaw 를 가진 직육면체)
            ax.add_patch(mp.Rectangle((gx - gw / 2, gy - gl / 2), gw, gl,
                                      angle=deg, rotation_point=(gx, gy),
                                      fc=CB["footing"], alpha=0.38, hatch="xxx",
                                      ec=CB["footing"], lw=1.0, zorder=2))
            ax.text(gx, gy - 0.75, f"{gtop*100:.0f}cm", ha="center", va="top",
                    fontsize=6, color="#641E16", zorder=3)
        # 핑거 2개
        b = C.BerthCfg()
        off = b.pile_gap / 2 + b.pile_diameter / 2
        for sgn in (-1, 1):
            ax.add_patch(mp.Rectangle((bx + sgn * off - b.pile_diameter / 2,
                                       G.WALL_Y - b.pile_length),
                                      b.pile_diameter, b.pile_length,
                                      fc=CB["finger"], ec="#7D6608", lw=1.0, zorder=4))
        # 목표
        ax.plot(tgt[0], tgt[1], "+", color="#145A32", ms=13, mew=2.2, zorder=5)

        # 유속 표시
        cu = float(d["current_u"])
        if abs(cu) > 1e-3:
            for yy in np.linspace(G.WALL_Y - 12.5, G.WALL_Y - 4.5, 4):
                x0 = bx - 6.5 if cu > 0 else bx + 6.5
                ax.annotate("", xy=(x0 + np.sign(cu) * 2.0, yy), xytext=(x0, yy),
                            arrowprops=dict(arrowstyle="-|>", color="#5DADE2",
                                            lw=1.6, alpha=0.7), zorder=1)
            ax.text(bx, G.WALL_Y - 13.0, f"current {cu:+.2f} m/s",
                    ha="center", fontsize=7.5, color="#2471A3")

        ax.set_title(title, fontsize=10, pad=4)

        # --- 동적 요소 ---
        self.lid_pts, = ax.plot([], [], ".", color=CB["lidar"], ms=3.2, zorder=6)
        self.fls_pts, = ax.plot([], [], ".", color=CB["fls"], ms=4.0, zorder=6)
        self.fls_fan = mp.Polygon(np.zeros((3, 2)), closed=True, fc=CB["fls"],
                                  alpha=0.09, zorder=2)
        ax.add_patch(self.fls_fan)
        self.trail, = ax.plot([], [], "-", color=CB["trail"], lw=1.2, alpha=0.65, zorder=5)
        self.det, = ax.plot([], [], "*", color=CB["det"], ms=13, zorder=9)

        self.hulls = [mp.Polygon(np.zeros((5, 2)), closed=True, fc=CB["hull"],
                                 ec=CB["hull_e"], lw=0.9, zorder=8) for _ in range(2)]
        self.deck = mp.Polygon(np.zeros((4, 2)), closed=True, fc=CB["deck"],
                               ec=CB["hull_e"], lw=0.7, zorder=7)
        for h in self.hulls:
            ax.add_patch(h)
        ax.add_patch(self.deck)
        self.arrow = ax.annotate("", xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle="-|>", color=CB["hull_e"], lw=2.0),
                                 zorder=9)
        # 화면 밖 표시 — 초기 정책은 20 m 이상 표류한다. 뷰를 넓히면 버스 detail 이
        # 안 보이므로, 뷰는 버스 주변에 고정하고 밖으로 나가면 가장자리에 방향 표식을 둔다.
        self.offmark, = ax.plot([], [], "v", color=CB["trail"], ms=11, zorder=10)
        self.offtxt = ax.text(0, 0, "", fontsize=7.5, color=CB["trail"],
                              ha="center", va="top", zorder=10)
        self.hud = ax.text(0.02, 0.975, "", transform=ax.transAxes, va="top", ha="left",
                           fontsize=7.5, family="monospace",
                           bbox=dict(fc="white", alpha=0.82, ec="none", pad=2.5), zorder=10)
        self.verdict = ax.text(0.33, 0.30, "", transform=ax.transAxes, ha="center",
                               fontsize=12, fontweight="bold", zorder=11)

        # ── 서브피겨: 센서가 "보는" 것 ──────────────────────────────────────
        # 메인 뷰는 관측자 시점(세계좌표)이고, 인셋은 **선체 고정 센서 시점**이다.
        # 둘을 같이 보여야 "정책이 무엇을 근거로 움직였는가"가 읽힌다.
        sm = C.SensorMountCfg()

        # LiDAR — 360° 극좌표. 선수가 위(0°)를 향하도록 그린다.
        self.ax_lid = ax.inset_axes([0.655, 0.345, 0.32, 0.32], projection="polar")
        a = self.ax_lid
        a.set_theta_zero_location("N"); a.set_theta_direction(1)
        a.set_ylim(0, sm.lidar_obs_max_range)
        a.set_yticks([]); a.set_xticks(np.linspace(0, 2 * np.pi, 4, endpoint=False))
        a.set_xticklabels(["bow", "P", "", "S"], fontsize=5.5)
        a.grid(alpha=0.25, lw=0.4); a.set_facecolor("#F4F8FB")
        a.set_title("LiDAR 360°", fontsize=6.5, pad=1.5)
        self.lid_polar, = a.plot([], [], ".", color=CB["lidar"], ms=1.8)
        # 검출된 버스 중심 — "정책이 무엇을 보고 판단했는가"를 센서 시점에서도 보여준다
        self.lid_star, = a.plot([], [], "*", color=CB["det"], ms=9, zorder=5)

        # FLS — 전방 부채꼴만.
        self.ax_fls = ax.inset_axes([0.655, 0.055, 0.32, 0.26], projection="polar")
        b2 = self.ax_fls
        half = np.radians(sm.fls_h_fov_deg / 2)
        b2.set_theta_zero_location("N"); b2.set_theta_direction(1)
        b2.set_thetamin(-sm.fls_h_fov_deg / 2); b2.set_thetamax(sm.fls_h_fov_deg / 2)
        b2.set_ylim(0, sm.fls_max_range)
        b2.set_yticks([]); b2.set_xticks([-half, 0, half])
        b2.set_xticklabels([f"-{sm.fls_h_fov_deg/2:.0f}°", "bow", f"+{sm.fls_h_fov_deg/2:.0f}°"],
                           fontsize=5.5)
        b2.grid(alpha=0.25, lw=0.4); b2.set_facecolor("#F2FBF5")
        b2.set_title("FLS fan (underwater)", fontsize=6.5, pad=1.5)
        self.fls_polar, = b2.plot([], [], ".", color=CB["fls"], ms=2.6)

        # 속력 게이지 + 추진기 2개 — 감속을 눈으로 확인하는 장치
        self.ax_g = ax.inset_axes([0.035, 0.055, 0.34, 0.135])
        g = self.ax_g
        g.set_xlim(0, 1); g.set_ylim(-0.5, 2.6)
        g.set_xticks([]); g.set_yticks([]); g.set_facecolor("#FFFFFF"); g.patch.set_alpha(0.85)
        for sp in g.spines.values():
            sp.set_linewidth(0.5)
        # 속력 바 (0 ~ 최대속력). 도킹 허용속력에 기준선을 둔다.
        self.bar_spd = g.barh([2.0], [0], height=0.7, color="#2E86C1")[0]
        g.axvline(0.20 / C.MAX_SPEED, color="#1E8449", lw=1.2, ls="--")
        g.text(0.20 / C.MAX_SPEED + 0.015, 2.0, "dock tol", fontsize=5, va="center",
               color="#1E8449")
        g.text(-0.01, 2.0, "spd", fontsize=5.5, ha="right", va="center")
        # 추진기 2개: 중앙 0.5 기준 좌우로 뻗는 막대(전진 초록 / 후진 빨강)
        self.bar_p = g.barh([1.0], [0], left=0.5, height=0.55, color="#27AE60")[0]
        self.bar_s = g.barh([0.0], [0], left=0.5, height=0.55, color="#27AE60")[0]
        g.axvline(0.5, color="#566573", lw=0.7)
        g.text(-0.01, 1.0, "port", fontsize=5.5, ha="right", va="center")
        g.text(-0.01, 0.0, "stbd", fontsize=5.5, ha="right", va="center")

    def _boat_polys(self, x, y, yaw):
        L, B = C.LOA, C.BEAM
        hw = 0.22
        c, s = math.cos(yaw), math.sin(yaw)

        def T(px, py):
            return (x + px * c - py * s, y + px * s + py * c)

        polys = []
        for sgn in (-1, 1):
            yc = sgn * (B / 2 - hw / 2)
            pts = [(-L / 2, yc - hw / 2), (L / 2 - 0.15, yc - hw / 2), (L / 2, yc),
                   (L / 2 - 0.15, yc + hw / 2), (-L / 2, yc + hw / 2)]
            polys.append(np.array([T(*p) for p in pts]))
        deck = np.array([T(*p) for p in [(-0.2, -B / 2), (0.2, -B / 2),
                                         (0.2, B / 2), (-0.2, B / 2)]])
        return polys, deck, T(L / 2, 0), T(L / 2 + 0.9, 0)

    def update(self, k):
        d = self.d
        k = min(k, self.n - 1)
        x, y = d["pos"][k]
        yaw = float(d["yaw"][k])

        x0, x1 = self.ax.get_xlim()
        y0, y1 = self.ax.get_ylim()
        inside = (x0 < x < x1) and (y0 < y < y1)

        hp, deck, tip, tipe = self._boat_polys(x, y, yaw)
        for poly, pts in zip(self.hulls, hp):
            poly.set_xy(pts if inside else np.zeros((5, 2)))
        self.deck.set_xy(deck if inside else np.zeros((4, 2)))
        if inside:
            self.arrow.set_position(tip)
            self.arrow.xy = tipe
        else:
            self.arrow.set_position((x0, y0))
            self.arrow.xy = (x0, y0)
            cx = min(max(x, x0 + 0.6), x1 - 0.6)
            cy = min(max(y, y0 + 0.6), y1 - 0.6)
            self.offmark.set_data([cx], [cy])
            self.offtxt.set_position((cx, cy - 0.35))
            self.offtxt.set_text(f"off view\n{math.hypot(x - cx, y - cy):.0f} m")
        if inside:
            self.offmark.set_data([], [])
            self.offtxt.set_text("")

        # LiDAR 반사점
        s = C.SensorMountCfg()
        lid = d["lidar"][k]
        R = len(lid)
        az = np.arange(R) * (2 * math.pi / R) + yaw
        rng = lid * s.lidar_obs_max_range
        m = lid < 0.999
        self.lid_pts.set_data(x + rng[m] * np.cos(az[m]), y + rng[m] * np.sin(az[m]))

        # FLS 부채꼴 + 반사점
        fl = d["fls"][k]
        nb = len(fl)
        half = math.radians(s.fls_h_fov_deg / 2)
        fang = yaw + np.linspace(-half, half, nb)
        frng = fl * s.fls_max_range
        fx, fy = x + frng * np.cos(fang), y + frng * np.sin(fang)
        self.fls_fan.set_xy(np.vstack([[x, y], np.column_stack([fx, fy])]))
        fm = fl < 0.999
        self.fls_pts.set_data(fx[fm], fy[fm])

        self.trail.set_data(d["pos"][: k + 1, 0], d["pos"][: k + 1, 1])

        if bool(d["det_ok"][k]):
            self.det.set_data([x + float(d["det_x"][k])], [y + float(d["det_wall"][k])])
        else:
            self.det.set_data([], [])

        u, v, r = d["nu"][k]
        dist = float(np.linalg.norm(d["pos"][k] - d["target"][:2]))
        # 정책 원시 출력은 범위를 벗어날 수 있다. 환경이 내부에서 [-1,1] 로 clamp 하므로
        # 실제 가해진 값을 표시해야 한다(원시값을 찍으면 +10 같은 숫자가 나온다).
        ap, as_ = np.clip(d["action"][k], -1.0, 1.0)
        self.hud.set_text(
            f"t   {k*self.dt:5.1f} s\n"
            f"dist{dist:6.2f} m\n"
            f"spd {math.hypot(u, v):5.2f} m/s"
        )
        # --- 서브피겨 갱신 (선체 고정 시점) ---
        az_b = np.arange(R) * (2 * math.pi / R)  # 선체 기준 방위
        self.lid_polar.set_data(az_b[m], rng[m])
        fang_b = np.linspace(-half, half, nb)
        self.fls_polar.set_data(fang_b[fm], frng[fm])

        # 검출된 버스 중심을 선체 고정 극좌표로 변환해 LiDAR 인셋에 찍는다
        if bool(d["det_ok"][k]):
            dxw, dyw = float(d["det_x"][k]), float(d["det_wall"][k])
            cy_, sy_ = math.cos(yaw), math.sin(yaw)
            bxx = dxw * cy_ + dyw * sy_          # 선체 전방
            byy = -dxw * sy_ + dyw * cy_         # 선체 좌현
            self.lid_star.set_data([math.atan2(byy, bxx)], [math.hypot(bxx, byy)])
        else:
            self.lid_star.set_data([], [])

        # --- 게이지 ---
        spd = math.hypot(u, v)
        self.bar_spd.set_width(min(spd / C.MAX_SPEED, 1.0))
        for bar, val in ((self.bar_p, ap), (self.bar_s, as_)):
            bar.set_width(0.5 * val)          # 중앙 0.5 에서 좌우로
            bar.set_color("#27AE60" if val >= 0 else "#C0392B")

        if k >= self.n - 1:
            self.verdict.set_text(OUTCOME_KO[self.outcome])
            self.verdict.set_color(OUTCOME_COL[self.outcome])
        else:
            self.verdict.set_text("")
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--titles", nargs="*", default=None)
    # 기본 출력은 figures/latest — make_figures.py 가 갱신하는 심볼릭 링크를 따라간다.
    # 시나리오가 바뀔 때마다 폴더를 새로 만들고 링크만 옮기면 이전 결과가 보존된다.
    ap.add_argument("--out", default=f"{ROOT}/figures/latest/learning_progress.gif")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=2, help="프레임 솎기(속도)")
    a = ap.parse_args()

    data = [np.load(f, allow_pickle=True) for f in a.npz]
    titles = a.titles or [os.path.basename(f).replace(".npz", "") for f in a.npz]

    n = len(data)
    rows, cols = (1, n) if n <= 2 else (2, (n + 1) // 2)
    fig, axes = plt.subplots(rows, cols, figsize=(7.4 * cols, 6.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    panels = [Panel(axes[i], d, titles[i]) for i, d in enumerate(data)]
    for j in range(n, len(axes)):
        axes[j].axis("off")

    total = max(p.n for p in panels)
    frames = list(range(0, total, a.stride)) + [total - 1] * (a.fps * 2)  # 끝에서 2초 정지

    fig.suptitle("BlueBoat autonomous docking — policy behaviour across training",
                 fontsize=13, y=0.985)
    # 범례
    fig.text(0.5, 0.012,
             "● LiDAR returns (above water)    ● FLS returns (underwater)    "
             "★ detected berth centre    ✚ target    ▨ derelict gear (LiDAR-invisible)    ⌐ ⌐ safety margin 0.5 m",
             ha="center", fontsize=8.5, color="#34495E")

    def upd(k):
        out = []
        for p in panels:
            out += p.update(k)
        return out

    anim = animation.FuncAnimation(fig, upd, frames=frames, interval=1000 // a.fps, blit=False)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    if a.out.endswith(".mp4"):
        try:
            anim.save(a.out, writer=animation.FFMpegWriter(fps=a.fps, bitrate=3200))
        except Exception as e:  # noqa: BLE001
            print(f"  ffmpeg 실패({e}) → GIF 로 대체", flush=True)
            a.out = a.out.replace(".mp4", ".gif")
            anim.save(a.out, writer=animation.PillowWriter(fps=a.fps))
    else:
        anim.save(a.out, writer=animation.PillowWriter(fps=a.fps))
    plt.close(fig)
    print(f"저장: {a.out}  ({len(frames)} 프레임)")

    for p, t in zip(panels, titles):
        print(f"  {t:28} {p.outcome:14} {p.n*p.dt:5.1f}s")


if __name__ == "__main__":
    main()
