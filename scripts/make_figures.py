"""발표용 그림 생성 (matplotlib, Isaac Sim 불필요).

만드는 그림:
  fig1_scenario.png   시나리오 전경 — 수조·버스·유속·초기 배치
  fig2_sensors.png    센서 시야 — LiDAR/FLS 가 각각 무엇을 보는가 (상보성)
  fig3_lidar_fov.png  Mid-360 수직시야가 마스트 높이에 따라 만드는 사각
  fig4_scan.png       실제 스캔 한 장면 — 광선, 반사점, 검출된 버스
  fig5_limit.png      정횡 유지 한계유속 (0.479 m/s) 과 PID 실측

영문 라벨을 쓴다 — 발표자료가 영문이기 때문이다(공고문 명시).
"""

import math
import sys
import importlib.util
import types

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = "/home/jason/07_USV_Docking_IsaacSim"


def _load(n, p):
    spec = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(spec)
    sys.modules[n] = m
    spec.loader.exec_module(m)
    return m


pkg = types.ModuleType("usvdock")
pkg.__path__ = [f"{ROOT}/usvdock"]
sys.modules["usvdock"] = pkg
C = _load("usvdock.blueboat_cfg", f"{ROOT}/usvdock/blueboat_cfg.py")
G = _load("usvdock.geometry", f"{ROOT}/usvdock/geometry.py")
P = _load("usvdock.perception", f"{ROOT}/usvdock/perception.py")

import argparse
import os
from datetime import datetime

# 그림을 덮어쓰지 않는다. 시나리오·설정이 바뀔 때마다 이전 그림과 비교할 수 있어야 한다.
#   --tag 를 주면 figures/<tag>/ 에, 안 주면 figures/<날짜시각>/ 에 저장하고
#   figures/latest 심볼릭 링크를 갱신한다.
_ap = argparse.ArgumentParser()
_ap.add_argument("--tag", type=str, default=None)
_args, _ = _ap.parse_known_args()
_tag = _args.tag or datetime.now().strftime("%y%m%d_%H%M%S")
OUT = f"{ROOT}/figures/{_tag}"
os.makedirs(OUT, exist_ok=True)
_link = f"{ROOT}/figures/latest"
if os.path.islink(_link) or os.path.exists(_link):
    os.remove(_link)
os.symlink(OUT, _link)

CB = dict(hull="#E8833A", lidar="#2E86C1", fls="#27AE60", wall="#7F8C8D",
          finger="#B7950B", gear="#922B21", water="#D6EAF8", grid="#BDC3C7")
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 150, "savefig.bbox": "tight"})


# ---------------------------------------------------------------- 선체 그리기
def draw_boat(ax, x, y, yaw, scale=1.0, label=True):
    """BlueBoat 를 위에서 본 형상. 쌍동선이므로 선체 2개 + 데크.

    전방을 화살표로 표시한다 — 부족구동(횡추력 없음)이 이 연구의 핵심이라
    "어디를 향하고 있는가"가 그림에서 즉시 읽혀야 한다.
    """
    L, B = C.LOA * scale, C.BEAM * scale
    hull_w = 0.22 * scale
    c, s = math.cos(yaw), math.sin(yaw)

    def T(px, py):
        return (x + px * c - py * s, y + px * s + py * c)

    # 쌍동선 선체 2개
    for sgn in (-1, 1):
        yc = sgn * (B / 2 - hull_w / 2)
        pts = [(-L / 2, yc - hull_w / 2), (L / 2 - 0.15 * scale, yc - hull_w / 2),
               (L / 2, yc), (L / 2 - 0.15 * scale, yc + hull_w / 2), (-L / 2, yc + hull_w / 2)]
        ax.add_patch(mp.Polygon([T(*p) for p in pts], closed=True,
                                fc=CB["hull"], ec="#7E5109", lw=1.0, zorder=5))
    # 데크(크로스튜브)
    ax.add_patch(mp.Polygon([T(*p) for p in [(-0.2 * scale, -B / 2), (0.2 * scale, -B / 2),
                                             (0.2 * scale, B / 2), (-0.2 * scale, B / 2)]],
                            closed=True, fc="#F5CBA7", ec="#7E5109", lw=0.8, zorder=4))
    # 전방 화살표
    ax.annotate("", xy=T(L / 2 + 0.5 * scale, 0), xytext=T(L / 2, 0),
                arrowprops=dict(arrowstyle="-|>", color="#7E5109", lw=1.6), zorder=6)
    # 추진기 2개 (선미, 차동)
    for sgn in (-1, 1):
        px, py = T(C.THRUSTER_X * scale, sgn * C.THRUSTER_Y * scale)
        ax.plot(px, py, "s", ms=3.5 * scale, color="#34495E", zorder=6)
    if label:
        ax.plot([], [], "s", color="#34495E", ms=4, label="M200 thruster (differential)")


def draw_berth(ax, bx=0.0, gear=None):
    """버스(핑거 2개) + 선택적으로 폐어구.

    gear: [(x, y, w, l, yaw_deg), ...]  — yaw 를 가진 직육면체
    """
    b = C.BerthCfg()
    off = b.pile_gap / 2 + b.pile_diameter / 2
    for sgn in (-1, 1):
        ax.add_patch(mp.Rectangle((bx + sgn * off - b.pile_diameter / 2,
                                   G.WALL_Y - b.pile_length),
                                  b.pile_diameter, b.pile_length,
                                  fc=CB["finger"], ec="#7D6608", lw=1.0, zorder=3))
    if gear:
        m = b.gear_safety_margin
        for gx, gy, gw, gl, gd in gear:
            # 안전 이격 경계 — 정책은 이 경계를 따라 최단으로 지나가야 한다
            ax.add_patch(mp.Rectangle((gx - gw / 2 - m, gy - gl / 2 - m),
                                      gw + 2 * m, gl + 2 * m, angle=gd,
                                      rotation_point=(gx, gy), fc="none",
                                      ec=CB["gear"], ls="--", lw=1.0, alpha=0.6, zorder=1))
            ax.add_patch(mp.Rectangle((gx - gw / 2, gy - gl / 2), gw, gl, angle=gd,
                                      rotation_point=(gx, gy), fc=CB["gear"],
                                      alpha=0.40, hatch="xxx", ec=CB["gear"],
                                      lw=1.0, zorder=2))


def tank_axes(ax, xlim=None, ylim=None):
    t = C.TankCfg()
    ax.add_patch(mp.Rectangle((-t.length / 2, -t.width / 2), t.length, t.width,
                              fc=CB["water"], ec=CB["wall"], lw=2.0, zorder=0))
    ax.set_aspect("equal")
    ax.set_xlim(xlim or (-t.length / 2 - 1, t.length / 2 + 1))
    ax.set_ylim(ylim or (-t.width / 2 - 1, t.width / 2 + 1))
    ax.set_xlabel("x [m]  (along tank / flow direction)")
    ax.set_ylabel("y [m]")


# ================================================================= Fig 1
def fig_scenario():
    t = C.TankCfg()
    fig, ax = plt.subplots(figsize=(9, 6))
    tank_axes(ax)
    draw_berth(ax, 0.0, gear=[(-1.6, 4.6, 0.9, 0.6, 35), (1.9, 6.0, 0.8, 0.55, 110)])

    # 유속 화살표
    for yy in np.linspace(-8, 6, 5):
        ax.annotate("", xy=(-10, yy), xytext=(-4, yy),
                    arrowprops=dict(arrowstyle="-|>", color="#5DADE2", lw=2, alpha=0.75))
    ax.text(-7, 7.6, "Cross-flow  0 – 0.45 m/s", color="#2471A3", fontsize=11, ha="center")

    # 초기 배치 부채꼴
    for d in (6, 9, 12):
        th = np.linspace(-35, 35, 60) * math.pi / 180
        ax.plot(d * np.sin(th), G.WALL_Y - 2.5 - d * np.cos(th), "--",
                color="#616A6B", lw=0.8, alpha=0.8)
    ax.text(0, G.WALL_Y - 2.5 - 13.2, "initial pose sampling\n6–12 m,  ±35° bearing",
            ha="center", fontsize=9, color="#616A6B")

    draw_boat(ax, 3.5, -1.0, math.radians(115), scale=2.2)
    ax.set_title("BlueBoat autonomous docking — 35 × 20 m circulating water channel")
    ax.plot([], [], color=CB["finger"], lw=6, label="berth fingers (0.5 m above water)")
    ax.plot([], [], color=CB["gear"], lw=6, alpha=0.5,
            label="derelict fishing gear (LiDAR-invisible) + 0.5 m safety margin")
    ax.legend(loc="lower right", fontsize=8)
    fig.savefig(f"{OUT}/fig1_scenario.png")
    plt.close(fig)
    print("  fig1_scenario.png")


# ================================================================= Fig 2
def fig_sensors():
    """센서 시야 상보성 — 이 연구의 핵심 논지를 한 장으로."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    s = C.SensorMountCfg()
    bx, by, yaw = 0.4, 5.2, math.radians(80)

    for ax, mode in zip(axes, ("lidar", "fls")):
        tank_axes(ax, xlim=(-5, 5), ylim=(2, 11))
        draw_berth(ax, 0.0, gear=[(0.3, 7.0, 0.9, 0.6, 25)])
        draw_boat(ax, bx, by, yaw, scale=1.0, label=False)

        if mode == "lidar":
            th = np.linspace(0, 2 * math.pi, 300)
            r = 12
            ax.fill(bx + r * np.cos(th), by + r * np.sin(th), color=CB["lidar"],
                    alpha=0.10, zorder=1)
            ax.set_title("LiDAR — Livox Mid-360\n"
                         "360° H,  −7°…+52° V,  sees ONLY above-water solids")
            ax.plot([], [], color=CB["lidar"], lw=6, alpha=0.4, label="LiDAR coverage")
            # 수중 기초부는 보이지 않음을 명시
            ax.text(0.3, 7.0 - 1.1, "derelict gear\nINVISIBLE to LiDAR",
                    ha="center", va="center", fontsize=9, color="white",
                    bbox=dict(fc=CB["gear"], ec="none", alpha=0.85, pad=3), zorder=8)
        else:
            half = math.radians(s.fls_h_fov_deg / 2)
            th = np.linspace(yaw - half, yaw + half, 80)
            r = s.fls_max_range
            xs = np.concatenate([[bx], bx + r * np.cos(th)])
            ys = np.concatenate([[by], by + r * np.sin(th)])
            ax.fill(xs, ys, color=CB["fls"], alpha=0.16, zorder=1)
            ax.set_title("FLS — forward-looking sonar\n"
                         f"{s.fls_h_fov_deg:.0f}° H fan,  underwater only")
            ax.plot([], [], color=CB["fls"], lw=6, alpha=0.4, label="FLS coverage")
            ax.text(0.3, 7.0 - 1.1, "derelict gear\nDETECTED by FLS",
                    ha="center", va="center", fontsize=9, color="white",
                    bbox=dict(fc=CB["fls"], ec="none", alpha=0.9, pad=3), zorder=8)
        ax.legend(loc="lower left", fontsize=8)

    fig.suptitle("Complementary blind zones — the docking terminal phase lies in the LiDAR blind zone",
                 fontsize=12)
    fig.savefig(f"{OUT}/fig2_sensors.png")
    plt.close(fig)
    print("  fig2_sensors.png")


# ================================================================= Fig 3
def fig_lidar_fov():
    """Mid-360 수직시야 -7° 가 마스트 높이에 따라 만드는 사각. 설계 근거 그림."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    pile_top = C.BerthCfg().pile_height_above_water
    DOWN = math.radians(7.0)

    # (a) 측면 기하
    for h, col, ls in ((0.45, "#27AE60", "-"), (0.70, "#E67E22", "--"), (1.20, "#C0392B", ":")):
        d = np.linspace(0.2, 8, 200)
        z = h - d * math.tan(DOWN)
        ax1.plot(d, z, ls, color=col, lw=1.8, label=f"mast {h:.2f} m — lower FOV edge")
        if h > pile_top:
            dmin = (h - pile_top) / math.tan(DOWN)
            ax1.plot([dmin], [pile_top], "o", color=col, ms=7)
            ax1.annotate(f"{dmin:.2f} m", (dmin, pile_top), textcoords="offset points",
                         xytext=(4, 8), color=col, fontsize=9)
    ax1.axhline(pile_top, color=CB["finger"], lw=2.5, label=f"finger top ({pile_top} m)")
    ax1.axhline(0, color="#2E86C1", lw=1.2)
    ax1.set_xlabel("horizontal range [m]")
    ax1.set_ylabel("height above water [m]")
    ax1.set_title("(a) Mid-360 lower FOV edge (−7°)\nmast above finger top → finger drops out of view")
    ax1.set_ylim(-0.3, 1.4)
    ax1.legend(fontsize=8)

    # (b) 실측 최소 검출거리
    gaps = np.array([3.0, 1.5, 0.6, 0.2])
    meas = {0.45: [3.11, 1.73, 1.04, 0.83], 0.70: [3.11, 1.73, 1.89, 1.89],
            1.20: [5.50, 4.00, 3.10, 2.70]}
    expect = np.hypot(0.8, gaps)
    ax2.plot(gaps, expect, "k--", lw=1.4, label="expected (finger corner)")
    for h, col in ((0.45, "#27AE60"), (0.70, "#E67E22"), (1.20, "#C0392B")):
        ax2.plot(gaps, meas[h], "o-", color=col, lw=1.8, ms=6, label=f"mast {h:.2f} m")
    ax2.invert_xaxis()
    ax2.set_xlabel("gap to finger tip [m]  (approaching →)")
    ax2.set_ylabel("min LiDAR return range [m]")
    ax2.set_title("(b) Simulated: value jumps to the far wall\nwhen the finger is lost")
    ax2.legend(fontsize=8)
    fig.savefig(f"{OUT}/fig3_lidar_fov.png")
    plt.close(fig)
    print("  fig3_lidar_fov.png")


# ================================================================= Fig 4
def fig_scan():
    """실제 스캔 한 장면: 광선, 반사점, 검출 결과."""
    torch.manual_seed(4)
    p = G.sample_scene(1, "cpu", mission=2, current_ms=0.0)
    p.berth_x[:] = 0.0
    p.gear_on[:] = False
    p.gear_x[:, 0], p.gear_y[:, 0] = 0.6, 6.2
    p.gear_w[:, 0], p.gear_l[:, 0] = 0.95, 0.6
    p.gear_yaw[:, 0] = math.radians(30)
    p.gear_top[:, 0], p.gear_bot[:, 0] = 0.10, 2.1
    p.gear_on[:, 0] = True
    pos = torch.tensor([[1.1, 3.2]])
    yaw = torch.tensor([math.radians(72.0)])

    s = C.SensorMountCfg()
    R = s.lidar_det_h_bins
    scan = G.lidar_scan(pos, yaw, p, n_bins=R, mount=s)
    rng = (scan * s.lidar_obs_max_range)[0].numpy()
    az = np.arange(R) * (2 * math.pi / R)
    ang = float(yaw) + az
    px, py = float(pos[0, 0]) + rng * np.cos(ang), float(pos[0, 1]) + rng * np.sin(ang)
    hit = scan[0].numpy() < 0.999

    fl = G.fls_scan(pos, yaw, p)
    nb = s.fls_n_beams
    frng = (fl * s.fls_max_range)[0].numpy()
    fang = float(yaw) + np.linspace(-math.radians(s.fls_h_fov_deg / 2),
                                    math.radians(s.fls_h_fov_deg / 2), nb)
    fx, fy = float(pos[0, 0]) + frng * np.cos(fang), float(pos[0, 1]) + frng * np.sin(fang)
    fhit = fl[0].numpy() < 0.999

    rel_x, wall_d, ok = P.detect_berth(pos, yaw, p)

    fig, ax = plt.subplots(figsize=(9, 7))
    tank_axes(ax, xlim=(-6, 7), ylim=(1.5, 11))
    draw_berth(ax, 0.0, gear=[(0.6, 6.2, 0.95, 0.6, 30)])

    for i in range(0, R, 3):
        if hit[i]:
            ax.plot([pos[0, 0], px[i]], [pos[0, 1], py[i]], color=CB["lidar"],
                    lw=0.35, alpha=0.30, zorder=1)
    ax.plot(px[hit], py[hit], ".", color=CB["lidar"], ms=2.5, zorder=3,
            label="LiDAR returns (above water)")
    ax.plot(fx[fhit], fy[fhit], ".", color=CB["fls"], ms=3.5, zorder=3,
            label="FLS returns (underwater)")

    draw_boat(ax, float(pos[0, 0]), float(pos[0, 1]), float(yaw), scale=1.0, label=False)

    if bool(ok[0]):
        ex = float(pos[0, 0]) + float(rel_x[0])
        ey = float(pos[0, 1]) + float(wall_d[0])
        ax.plot(ex, ey, "*", color="#C0392B", ms=18, zorder=9,
                label=f"detected berth centre (err {abs(ex):.3f} m)")
        ax.plot([ex, ex], [ey - 2.5, ey], "-.", color="#C0392B", lw=1.2, zorder=8)

    ax.set_title("Single-frame perception — no SLAM, no mapping, no temporal filtering\n"
                 "berth centre extracted geometrically from one scan")
    ax.legend(loc="lower left", fontsize=8)
    fig.savefig(f"{OUT}/fig4_scan.png")
    plt.close(fig)
    print("  fig4_scan.png")


# ================================================================= Fig 5
def fig_limit():
    """부족구동 선박은 횡류 중 도킹 자세를 유지할 수 없다 — 정정된 물리.

    ※ 초기 그림은 "정횡 유지 한계유속 0.479 m/s" 를 그렸으나 그 계산은 틀렸다.
      도킹 자세에서 추력은 전부 종방향이라 횡력으로 쓸 수 없으므로
      '필요 횡력 vs 최대 추력' 비교가 성립하지 않는다.
      올바른 결론은 더 강하다: **90° 자세에서는 어떤 유속에서도 평형이 없다.**
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) 표류 시간
    v = np.linspace(0.02, 0.5, 200)
    a_lat = (60 * v + 180 * v**2) / 60.0  # 유효 횡질량 60 kg
    t_out = np.sqrt(2 * 0.25 / a_lat)
    ax1.plot(v, t_out, color="#2E4053", lw=2.2)
    ax1.axvspan(0, 0.45, color="#27AE60", alpha=0.10)
    for vv in (0.10, 0.20, 0.30, 0.45):
        tt = np.sqrt(2 * 0.25 / ((60 * vv + 180 * vv**2) / 60.0))
        ax1.plot([vv], [tt], "o", color="#C0392B", ms=6)
        ax1.annotate(f"{tt:.2f}s", (vv, tt), textcoords="offset points",
                     xytext=(5, 6), fontsize=8, color="#C0392B")
    ax1.axhline(0.5, color="#1E8449", ls="--", lw=1.5)
    ax1.text(0.30, 0.58, "success hold 0.5 s", fontsize=8, color="#1E8449")
    ax1.set_xlabel("beam-on current [m/s]")
    ax1.set_ylabel("time to drift out of 0.25 m tolerance [s]")
    ax1.set_title("(a) No equilibrium at 90° heading\n"
                  "underactuated hull cannot resist beam-on flow at all")
    ax1.set_ylim(0, 3)

    # (b) PID 실측 (이전 시나리오, 참고값)
    cur = [0.0, 0.5, 1.0, 1.75]
    sr = [95.3, 0.0, 0.0, 0.0]
    ax2.bar([f"{c:.2f}" for c in cur], sr,
            color=["#27AE60", "#C0392B", "#C0392B", "#C0392B"])
    ax2.set_xlabel("current [m/s]")
    ax2.set_ylabel("PID docking success rate [%]")
    ax2.set_title("(b) Measured PID baseline\n"
                  "study range 0–0.45 m/s set from this, not from a force limit")
    for i, s_ in enumerate(sr):
        ax2.text(i, s_ + 2, f"{s_:.1f}%", ha="center", fontsize=9)
    ax2.set_ylim(0, 108)
    fig.savefig(f"{OUT}/fig5_limit.png")
    plt.close(fig)
    print("  fig5_limit.png")


if __name__ == "__main__":
    print("그림 생성:")
    fig_scenario()
    fig_sensors()
    fig_lidar_fov()
    fig_scan()
    fig_limit()
    print(f"→ {OUT}/   (figures/latest 로도 접근 가능)")
