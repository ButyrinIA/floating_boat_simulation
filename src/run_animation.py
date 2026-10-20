"""Анимация плавучести с пересчётом силы по полигонам на каждом кадре."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.forces.buoyancy import BoatModel
from src.utils.config import load_config
from src.visualization.animation import _resize_window, plot_boat_projections


def main():
    config = load_config()
    boat = BoatModel(config["model_path"])

    rotation = config.get("rotation_deg", {})
    boat.rotate_boat(
        alpha=np.radians(rotation.get("alpha", 0)),
        psi=np.radians(rotation.get("psi", 0)),
        phi=np.radians(rotation.get("phi", 0)),
    )

    rho = config["fluid_density"]
    g = config["gravity"]
    position = np.zeros(3)
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])

    wl_min, wl_max = -0.8, 0.8
    n_frames = 80

    fig = plt.figure(figsize=(18, 10))
    ax_side = fig.add_subplot(2, 3, 1)
    ax_front = fig.add_subplot(2, 3, 2)
    ax_top = fig.add_subplot(2, 3, 3)
    ax_force = fig.add_subplot(2, 3, 4)
    ax_center = fig.add_subplot(2, 3, 5)
    ax_wl = fig.add_subplot(2, 3, 6)
    _resize_window(fig, 1800, 1000)

    times, forces_z, centers_z, water_levels = [], [], [], []

    def update(frame):
        wl = wl_min + (wl_max - wl_min) * frame / max(n_frames - 1, 1)
        boat.compute_buoyancy(water_level=wl, rho=rho, g=g)

        force = boat.get_total_buoyancy_force()
        center = boat.get_buoyancy_center()

        t = frame * 0.1
        times.append(t)
        forces_z.append(force[2])
        centers_z.append(center[2])
        water_levels.append(wl)

        plot_boat_projections(
            (ax_side, ax_front, ax_top), boat.mesh, position, quaternion, wl
        )

        ax_force.clear()
        ax_force.plot(times, forces_z, "m-")
        ax_force.set_title("Сила Архимеда (Z)")
        ax_force.set_xlabel("кадр")
        ax_force.set_ylabel("N")
        ax_force.grid(True)

        ax_center.clear()
        ax_center.plot(times, centers_z, "c-")
        ax_center.set_title("Точка приложения (Z)")
        ax_center.set_xlabel("кадр")
        ax_center.set_ylabel("м")
        ax_center.grid(True)

        ax_wl.clear()
        ax_wl.plot(times, water_levels, "b-")
        ax_wl.set_title("Уровень воды")
        ax_wl.set_xlabel("кадр")
        ax_wl.set_ylabel("м")
        ax_wl.grid(True)

        fig.suptitle(
            f"Кадр {frame}: wl={wl:.2f} м, |F|={np.linalg.norm(force):.0f} N",
            fontsize=11,
        )

    ani = FuncAnimation(fig, update, frames=n_frames, interval=150, repeat=True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
