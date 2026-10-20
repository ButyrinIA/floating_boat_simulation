import json
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "simulation.json"


def apply_overrides(config: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        parts = str(key).split(".")
        node = config
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child
        node[parts[-1]] = value
    return config


def load_raw_config(path: Path | str | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def save_raw_config(config: dict, path: Path | str | None = None) -> None:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
        f.write("\n")


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> dict:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    if overrides:
        apply_overrides(config, overrides)

    config["model_path"] = str(PROJECT_ROOT / config["model_path"])
    if "buoyancy_table_path" in config:
        config["buoyancy_table_path"] = str(PROJECT_ROOT / config["buoyancy_table_path"])
    return config


def grid_from_config(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = config["precompute_grid"]
    wl = grid["water_level"]
    alpha = grid["alpha_deg"]
    psi = grid["psi_deg"]

    wl_values = np.arange(wl["min"], wl["max"], wl["step"])
    alpha_values = np.arange(alpha["min"], alpha["max"] + alpha["step"], alpha["step"])
    psi_values = np.arange(psi["min"], psi["max"] + psi["step"], psi["step"])
    return wl_values, alpha_values, psi_values
