"""Предрасчёт таблицы плавучести."""

from src.forces.buoyancy import precompute_buoyancy_table
from src.utils.config import grid_from_config, load_config


def run(config: dict):
    wl_values, alpha_values, psi_values = grid_from_config(config)

    n_angles = len(alpha_values) * len(psi_values)
    n_wl = len(wl_values)
    print(f"Модель: {config['model_path']}")
    print(f"Сетка: {len(wl_values)} уровней воды × {n_angles} углов = {n_wl * n_angles} расчётов углов")
    print(f"Сохранение: {config['buoyancy_table_path']}")
    print("Это может занять часы — уменьшите precompute_grid в config/simulation.json для теста.")

    precompute_buoyancy_table(
        obj_file=config["model_path"],
        water_levels=wl_values,
        alpha_values=alpha_values,
        psi_values=psi_values,
        save_path=config["buoyancy_table_path"],
    )
    print("Готово.")


def main():
    run(load_config())


if __name__ == "__main__":
    main()
