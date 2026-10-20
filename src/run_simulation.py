"""Симуляция движения лодки по таблице плавучести."""

from pathlib import Path

import numpy as np
import trimesh
import matplotlib.pyplot as plt

from src.dynamics.boat_dynamics import (
    BoatParams,
    BuoyancyTableContext,
    drag_force_body,
    normalize_quaternion_in_state,
    state_derivative,
)
from src.utils.rotations import TRot3D
from src.utils.config import grid_from_config, load_config
from src.utils.table_loader import calculate_initial_equilibrium, load_buoyancy_table
from src.visualization.animation import animate_simulation, plot_simulation_history


def rk4_step(state, dt, params, buoyancy):
    k1 = state_derivative(state, params, buoyancy)
    k2 = state_derivative(state + 0.5 * dt * k1, params, buoyancy)
    k3 = state_derivative(state + 0.5 * dt * k2, params, buoyancy)
    k4 = state_derivative(state + dt * k3, params, buoyancy)
    new_state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return normalize_quaternion_in_state(new_state)


def buoyancy_magnitude(state, buoyancy_ctx):
    from src.dynamics.boat_dynamics import _angles_from_quaternion
    from src.utils.table_loader import triquadratic_interpolation

    Z = state[2]
    q0, q1, q2, q3 = state[3:7]
    alpha, psi = _angles_from_quaternion(q0, q1, q2, q3)
    _, force = triquadratic_interpolation(
        buoyancy_ctx.water_level - Z,
        alpha,
        psi,
        buoyancy_ctx.K_force,
        buoyancy_ctx.K_center,
        buoyancy_ctx.wl_values,
        buoyancy_ctx.alpha_values,
        buoyancy_ctx.psi_values,
    )
    return float(np.linalg.norm(force))


def run_integration(params: BoatParams, buoyancy_ctx: BuoyancyTableContext, sim_cfg: dict, z0: float):
    dt = sim_cfg["dt"]
    t_max = sim_cfg["t_max"]
    n_steps = int(t_max / dt) + 1

    q_init = sim_cfg.get("initial_quaternion", [1.0, 0.0, 0.0, 0.0])
    state = normalize_quaternion_in_state(np.array([
        0.0, 0.0, z0,
        q_init[0], q_init[1], q_init[2], q_init[3],
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ]))

    history = {
        "t": np.zeros(n_steps),
        "X": np.zeros(n_steps),
        "Y": np.zeros(n_steps),
        "Z": np.zeros(n_steps),
        "q0": np.zeros(n_steps),
        "q1": np.zeros(n_steps),
        "q2": np.zeros(n_steps),
        "q3": np.zeros(n_steps),
        "VX": np.zeros(n_steps),
        "VY": np.zeros(n_steps),
        "VZ": np.zeros(n_steps),
        "F_buoy": np.zeros(n_steps),
        "F_drag": np.zeros(n_steps),
    }

    for step in range(n_steps):
        history["t"][step] = step * dt
        history["X"][step] = state[0]
        history["Y"][step] = state[1]
        history["Z"][step] = state[2]
        history["q0"][step] = state[3]
        history["q1"][step] = state[4]
        history["q2"][step] = state[5]
        history["q3"][step] = state[6]
        history["VX"][step] = state[7]
        history["VY"][step] = state[8]
        history["VZ"][step] = state[9]
        history["F_buoy"][step] = buoyancy_magnitude(state, buoyancy_ctx)
        vx, vy, vz = TRot3D(state[7], state[8], state[9], state[3], state[4], state[5], state[6])
        immersion = buoyancy_ctx.water_level - state[2]
        history["F_drag"][step] = float(np.linalg.norm(drag_force_body(vx, vy, vz, immersion, params)))

        if step < n_steps - 1:
            state = rk4_step(state, dt, params, buoyancy_ctx)

    return history


def run(config: dict):
    sim_cfg = config.get("simulation", {})
    table_path = Path(config["buoyancy_table_path"])

    if not table_path.is_file():
        print(f"Нет таблицы: {table_path}")
        print("Сначала: python -m src.main precompute")
        return

    wl_values, alpha_values, psi_values = grid_from_config(config)
    try:
        K_force, K_center, wl_values, alpha_values, psi_values = load_buoyancy_table(
            table_path,
            wl_values,
            alpha_values,
            psi_values,
            scale=config.get("table_scale", 1.0),
        )
    except MemoryError as e:
        print(e)
        return

    water_level = config["water_level"]
    buoyancy_ctx = BuoyancyTableContext(
        K_force=K_force,
        K_center=K_center,
        wl_values=wl_values,
        alpha_values=alpha_values,
        psi_values=psi_values,
        water_level=water_level,
    )

    mass = config["mass"]
    mesh = trimesh.load(config["model_path"], process=False)
    extents = np.asarray(mesh.bounds[1] - mesh.bounds[0], dtype=float)
    rx, ry, rz = (float(v) for v in extents / 2.0)
    params = BoatParams(
        mass=mass,
        rho=config["fluid_density"],
        g=config["gravity"],
        R_x=rx,
        R_y=ry,
        R_z=rz,
        length=float(np.max(extents)),
        trust_body=tuple(sim_cfg.get("trust_force_body", [0.0, 20000.0, 0.0])),
    )
    ahead = [0.0, 0.0, 0.0]
    ahead[int(np.argmax((rx, ry)))] = 1.0
    drag_at_1 = drag_force_body(*ahead, 0.0, params)
    print(
        f"Эллипсоид: полуоси {rx:.3f}, {ry:.3f}, {rz:.3f} м, "
        f"S={params.wetted_area:.2f} м²"
    )
    print(
        f"На плаву, 1 м/с вдоль длины: сопротивление {np.linalg.norm(drag_at_1):.1f} Н "
        f"(Fx={drag_at_1[0]:.1f}, Fy={drag_at_1[1]:.1f}, Fz={drag_at_1[2]:.1f})"
    )

    if sim_cfg.get("use_equilibrium_z", True):
        immersion_eq = calculate_initial_equilibrium(
            mass, K_force, K_center, wl_values, alpha_values, psi_values, g=params.g
        )
        z0 = water_level - immersion_eq
        print(f"Начальное Z (равновесие): {z0:.4f} м (погружение {immersion_eq:.4f} м)")
    else:
        z0 = sim_cfg.get("initial_z", 0.0)
        print(f"Начальное Z (из конфига): {z0:.4f} м")

    print(f"Интегрирование: t_max={sim_cfg.get('t_max', 60)} с, dt={sim_cfg.get('dt', 0.1)} с")
    history = run_integration(params, buoyancy_ctx, sim_cfg, z0)

    plot_simulation_history(history, water_level)

    if sim_cfg.get("animate", True):
        _, ani = animate_simulation(
            history,
            mesh,
            water_level,
            stride=sim_cfg.get("animation_stride", 5),
        )
        plt.show()
    else:
        plt.show()


def main():
    run(load_config())


if __name__ == "__main__":
    main()
