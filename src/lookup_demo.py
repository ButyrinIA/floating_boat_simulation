"""Одна точка из таблицы плавучести, без динамики."""

import numpy as np

from src.utils.config import grid_from_config, load_config
from src.utils.table_loader import load_buoyancy_table, lookup_buoyancy


def main():
    config = load_config()
    table_path = config["buoyancy_table_path"]

    try:
        wl_values, alpha_values, psi_values = grid_from_config(config)
        K_force, K_center, wl_values, alpha_values, psi_values = load_buoyancy_table(
            table_path,
            wl_values,
            alpha_values,
            psi_values,
            scale=config.get("table_scale", 1.0),
        )
    except FileNotFoundError:
        print(f"Файл не найден: {table_path} / {table_path.replace('.pkl', '.npz')}")
        print("Сначала: py -3.14 -m src.precompute_table")
        return
    except MemoryError as e:
        print(e)
        return

    z = 0.0
    alpha = 0.0
    psi = 0.0
    center, force = lookup_buoyancy(
        z, alpha, psi,
        K_force, K_center, wl_values, alpha_values, psi_values,
        config["water_level"],
    )

    print(f"Lookup: z={z}, alpha={alpha}°, psi={psi}°")
    print(f"  Сила: {force}, |F|={np.linalg.norm(force):.1f} N")
    print(f"  Точка приложения: {center}")


if __name__ == "__main__":
    main()
