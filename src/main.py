"""Окно параметров и запуск сценария."""

import argparse

import matplotlib.pyplot as plt

from src.precompute_table import run as run_precompute
from src.run_mesh_view import run as run_mesh
from src.run_simulation import run as run_simulation
from src.utils.config import apply_overrides, load_config, load_raw_config, save_raw_config
from src.visualization.launcher import MODES, serve_launcher

MODE = "simulation"

CONFIG_OVERRIDES = {
    # "mass": 7000,
    # "simulation.dt": 0.1,
}


def main():
    mode_ids = [mode for mode, _label in MODES]
    parser = argparse.ArgumentParser(description="Окно параметров лодки и запуск сценария")
    parser.add_argument(
        "mode",
        nargs="?",
        choices=mode_ids,
        default=None,
        help="какой сценарий отметить при открытии окна",
    )
    args = parser.parse_args()

    raw = load_raw_config()
    if CONFIG_OVERRIDES:
        apply_overrides(raw, CONFIG_OVERRIDES)

    def run_choice(choice: dict):
        mode = choice.pop("mode")
        save_raw_config(choice)
        config = load_config()
        print(f"Режим: {mode}")
        try:
            if mode == "simulation":
                run_simulation(config)
            elif mode == "precompute":
                run_precompute(config)
            else:
                run_mesh(config, subdivide=int(config.get("mesh_subdivide", 0)))
        except KeyboardInterrupt:
            print("Сценарий прерван.")
        finally:
            plt.close("all")
        print("Сценарий завершён.")

    serve_launcher(raw, args.mode or MODE, run_choice)
    print("Окно параметров закрыто.")


if __name__ == "__main__":
    main()
