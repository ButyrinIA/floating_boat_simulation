"""Графики и анимация результатов симуляции."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.spatial.transform import Rotation as R
import trimesh


def _resize_window(fig, width: int, height: int) -> None:
    try:
        fig.canvas.manager.resize(width, height)
    except Exception:
        return


def world_vertices(mesh: trimesh.Trimesh, position, quaternion, scale: float = 1.0) -> np.ndarray:
    vertices = mesh.vertices * scale
    q = np.asarray(quaternion, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0])
    rot = R.from_quat([q[1], q[2], q[3], q[0]])
    pos = np.asarray(position, dtype=float)
    if not np.isfinite(pos).all():
        pos = np.zeros(3)
    return rot.apply(vertices) + pos


def plot_boat_projections(axes, mesh: trimesh.Trimesh, position, quaternion, water_level: float, scale: float = 1.0):
    """Сбоку, спереди и сверху в одном масштабе."""
    points = world_vertices(mesh, position, quaternion, scale)
    extents = np.ptp(np.asarray(mesh.vertices, dtype=float), axis=0) * scale
    half = 0.5 * float(np.max(extents)) + 2.0
    pos = np.asarray(position, dtype=float)
    if not np.isfinite(pos).all():
        pos = np.zeros(3)
    views = (
        (axes[0], 1, 2, "Сбоку (Y-Z)", "Y", "Z", True),
        (axes[1], 0, 2, "Спереди (X-Z)", "X", "Z", True),
        (axes[2], 0, 1, "Сверху (X-Y)", "X", "Y", False),
    )
    for ax, i, j, title, xlabel, ylabel, show_water in views:
        ax.clear()
        ax.plot(points[:, i], points[:, j], "k.", markersize=1)
        ax.set_xlim(pos[i] - half, pos[i] + half)
        ax.set_ylim(pos[j] - half, pos[j] + half)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)
        if show_water:
            ax.axhline(water_level, color="blue", linestyle="--", label="Уровень воды")
            ax.legend(loc="upper right", fontsize=8)


def plot_boat_side_view(
    ax,
    mesh: trimesh.Trimesh,
    position,
    quaternion,
    water_level: float,
    scale: float = 1.0,
    y_window: float = 15.0,
    z_lim=(-3.0, 3.0),
):
    points = world_vertices(mesh, position, quaternion, scale)
    pos = np.asarray(position, dtype=float)
    y = float(pos[1]) if np.isfinite(pos[1]) else 0.0
    ax.clear()
    ax.set_title("Сбоку (Y-Z)")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.set_xlim(-y_window + y, y_window + y)
    ax.set_ylim(z_lim)
    ax.set_aspect("equal")
    ax.grid(True)
    ax.plot(points[:, 1], points[:, 2], "k.", markersize=1)
    ax.axhline(y=water_level, color="blue", linestyle="--", label="Уровень воды")
    ax.legend(loc="upper right")


def plot_simulation_history(history: dict, water_level: float):
    """Графики положения, кватерниона, силы Архимеда и скорости."""
    t = history["t"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax = axes[0, 0]
    ax.plot(t, history["X"], "r-", label="X")
    ax.plot(t, history["Y"], "g-", label="Y")
    ax.plot(t, history["Z"], "b-", label="Z")
    ax.axhline(water_level, color="cyan", linestyle="--", alpha=0.5, label="уровень воды")
    ax.set_xlabel("t, с")
    ax.set_ylabel("м")
    ax.set_title("Положение")
    ax.legend()
    ax.grid(True)

    ax = axes[0, 1]
    ax.plot(t, history["q0"], label="q0")
    ax.plot(t, history["q1"], label="q1")
    ax.plot(t, history["q2"], label="q2")
    ax.plot(t, history["q3"], label="q3")
    ax.set_xlabel("t, с")
    ax.set_title("Кватернион")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 0]
    ax.plot(t, history["F_buoy"], "m-", label="|F_A|")
    if "F_drag" in history:
        ax.plot(t, history["F_drag"], "c-", label="|F_сопр|")
    ax.set_xlabel("t, с")
    ax.set_ylabel("N")
    ax.set_title("Сила Архимеда")
    ax.legend()
    ax.grid(True)

    ax = axes[1, 1]
    ax.plot(t, history["VX"], label="VX")
    ax.plot(t, history["VY"], label="VY")
    ax.plot(t, history["VZ"], label="VZ")
    ax.set_xlabel("t, с")
    ax.set_title("Скорость (связанная СК)")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return fig


def animate_simulation(
    history: dict,
    mesh: trimesh.Trimesh,
    water_level: float,
    stride: int = 5,
    interval_ms: int = 80,
):
    """Три проекции корпуса и нарастающие графики."""
    indices = list(range(0, len(history["t"]), max(1, stride)))
    t = history["t"]

    fig = plt.figure(figsize=(18, 10))
    ax_side = fig.add_subplot(2, 3, 1)
    ax_front = fig.add_subplot(2, 3, 2)
    ax_top = fig.add_subplot(2, 3, 3)
    ax_coords = fig.add_subplot(2, 3, 4)
    ax_buoy = fig.add_subplot(2, 3, 5)
    ax_q = fig.add_subplot(2, 3, 6)
    _resize_window(fig, 1800, 1000)

    def update(frame_idx):
        i = indices[frame_idx]
        pos = [history["X"][i], history["Y"][i], history["Z"][i]]
        quat = [history["q0"][i], history["q1"][i], history["q2"][i], history["q3"][i]]
        plot_boat_projections((ax_side, ax_front, ax_top), mesh, pos, quat, water_level)

        sl = slice(0, i + 1)
        ax_coords.clear()
        ax_coords.plot(t[sl], history["X"][sl], "r-", label="X")
        ax_coords.plot(t[sl], history["Y"][sl], "g-", label="Y")
        ax_coords.plot(t[sl], history["Z"][sl], "b-", label="Z")
        ax_coords.set_xlim(0, t[-1])
        ax_coords.set_title("Координаты")
        ax_coords.legend(fontsize=8)
        ax_coords.grid(True)

        ax_buoy.clear()
        ax_buoy.plot(t[sl], history["F_buoy"][sl], "m-", label="|F_A|")
        if "F_drag" in history:
            ax_buoy.plot(t[sl], history["F_drag"][sl], "c-", label="|F_сопр|")
        ax_buoy.set_xlim(0, t[-1])
        ax_buoy.set_title("Силы")
        ax_buoy.legend(fontsize=8)
        ax_buoy.grid(True)

        ax_q.clear()
        ax_q.plot(t[sl], history["q0"][sl], label="q0")
        ax_q.plot(t[sl], history["q1"][sl], label="q1")
        ax_q.plot(t[sl], history["q2"][sl], label="q2")
        ax_q.plot(t[sl], history["q3"][sl], label="q3")
        ax_q.set_xlim(0, t[-1])
        ax_q.set_title("Кватернион")
        ax_q.legend(fontsize=8)
        ax_q.grid(True)

        fig.suptitle(f"t = {t[i]:.1f} с, Z = {history['Z'][i]:.3f} м")

    ani = FuncAnimation(fig, update, frames=len(indices), interval=interval_ms, repeat=False)
    plt.tight_layout()
    return fig, ani
