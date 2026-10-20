"""Просмотр сетки и силы Архимеда по полигонам."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from src.forces.buoyancy import BoatModel
from src.utils.config import load_config
from src.visualization.buoyancy_view import plot_buoyancy


def run(config: dict, subdivide: int = 0):
    boat = BoatModel(config["model_path"])

    if subdivide > 0:
        boat.refine_mesh(subdivide)

    rotation = config.get("rotation_deg", {})
    boat.rotate_boat(
        alpha=np.radians(rotation.get("alpha", 0)),
        psi=np.radians(rotation.get("psi", 0)),
        phi=np.radians(rotation.get("phi", 0)),
    )

    water_level = config["water_level"]
    rho = config["fluid_density"]
    g = config["gravity"]
    boat.compute_buoyancy(water_level=water_level, rho=rho, g=g)

    total_force = boat.get_total_buoyancy_force()
    center = boat.get_buoyancy_center()
    magnitudes = np.array([np.linalg.norm(p.buoyancy_force) for p in boat.polygons])
    n_wet = int(np.count_nonzero(magnitudes))

    print(f"Файл модели: {config['model_path']}")
    print(f"Уровень воды: {water_level}")
    print(f"Полигонов: {len(boat.polygons)}, под водой: {n_wet}, над водой: {len(boat.polygons) - n_wet}")
    print(f"Суммарная сила Архимеда: {total_force}")
    print(f"Модуль силы: {np.linalg.norm(total_force):.2f} N")
    print(f"Точка приложения: {center}")

    plot_buoyancy(boat, water_level=water_level)
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Сетка модели и сила Архимеда по полигонам")
    parser.add_argument(
        "--subdivide",
        type=int,
        default=0,
        help="Число субдивизий trimesh (0 = как в OBJ, 1 = в 4 раза больше треугольников)",
    )
    args = parser.parse_args()
    run(load_config(), subdivide=args.subdivide)


if __name__ == "__main__":
    main()
