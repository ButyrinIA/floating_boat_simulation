"""Загрузка таблицы плавучести и интерполяция силы и центра давления."""

import pickle
from pathlib import Path

import numpy as np

MAX_PKL_BYTES = 100 * 1024 * 1024


def _grid_steps(wl_values, alpha_values, psi_values):
    wl_step = float(wl_values[1] - wl_values[0]) if len(wl_values) > 1 else 0.05
    alpha_step = float(alpha_values[1] - alpha_values[0]) if len(alpha_values) > 1 else 2.0
    psi_step = float(psi_values[1] - psi_values[0]) if len(psi_values) > 1 else 2.0
    return wl_step, alpha_step, psi_step


def table_dict_to_arrays(
    table: dict,
    wl_values: np.ndarray,
    alpha_values: np.ndarray,
    psi_values: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    wl_step, alpha_step, psi_step = _grid_steps(wl_values, alpha_values, psi_values)
    na, npsi, nwl = len(alpha_values), len(psi_values), len(wl_values)
    K_force = np.zeros((na, npsi, nwl, 3), dtype=np.float64)
    K_center = np.zeros((na, npsi, nwl, 3), dtype=np.float64)

    for (wl, alpha, psi), v in table.items():
        i = int(round((alpha - alpha_values[0]) / alpha_step))
        j = int(round((psi - psi_values[0]) / psi_step))
        k = int(round((wl - wl_values[0]) / wl_step))
        if 0 <= i < na and 0 <= j < npsi and 0 <= k < nwl:
            K_force[i, j, k] = np.asarray(v["force"], dtype=np.float64)
            K_center[i, j, k] = np.asarray(v["center"], dtype=np.float64) * scale

    return K_force, K_center


def save_table_npz(
    K_force: np.ndarray,
    K_center: np.ndarray,
    wl_values: np.ndarray,
    alpha_values: np.ndarray,
    psi_values: np.ndarray,
    path: str | Path,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        force=K_force,
        center=K_center,
        wl_values=np.asarray(wl_values, dtype=np.float64),
        alpha_values=np.asarray(alpha_values, dtype=np.float64),
        psi_values=np.asarray(psi_values, dtype=np.float64),
    )


def load_table_npz(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path)
    return (
        data["force"],
        data["center"],
        data["wl_values"],
        data["alpha_values"],
        data["psi_values"],
    )


def load_buoyancy_table(
    table_path: str | Path,
    wl_values: np.ndarray,
    alpha_values: np.ndarray,
    psi_values: np.ndarray,
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Сначала .npz рядом с путём таблицы, иначе небольшой .pkl."""
    base = Path(table_path)
    npz_path = base.with_suffix(".npz")
    if npz_path.is_file():
        K_force, K_center, wl, alpha, psi = load_table_npz(npz_path)
        if scale != 1.0:
            K_center = K_center * scale
        return K_force, K_center, wl, alpha, psi

    if not base.is_file():
        raise FileNotFoundError(f"Нет таблицы: {npz_path} или {base}")

    size = base.stat().st_size
    if size > MAX_PKL_BYTES:
        raise MemoryError(
            f"Файл {base} слишком большой ({size / 1e6:.0f} МБ). "
            f"Это старая таблица с огромной сеткой — в RAM не влезет.\n"
            f"Переименуйте/удалите его и создайте новую таблицу:\n"
            f"  py -3.14 -m src.precompute_table\n"
            f"(сетка задаётся в config/simulation.json → precompute_grid)"
        )

    with open(base, "rb") as f:
        raw = pickle.load(f)
    if not raw:
        raise ValueError("Таблица плавучести пуста.")

    K_force, K_center = table_dict_to_arrays(raw, wl_values, alpha_values, psi_values, scale)
    save_table_npz(K_force, K_center, wl_values, alpha_values, psi_values, npz_path)
    print(f"Создан компактный кэш: {npz_path}")
    return K_force, K_center, wl_values, alpha_values, psi_values


def load_table(table_path: str | Path, scale: float = 1.0) -> dict:
    with open(table_path, "rb") as f:
        raw = pickle.load(f)
    if not raw:
        raise ValueError("Таблица плавучести пуста.")
    return {
        k: {"force": np.array(v["force"]), "center": np.array(v["center"]) * scale}
        for k, v in raw.items()
    }


def load_table_to_3darray(
    table_path: str | Path,
    wl_values: np.ndarray,
    alpha_values: np.ndarray,
    psi_values: np.ndarray,
    scale: float = 1.0,
):
    """Устаревший формат: массив словарей вместо двух массивов силы и центра."""
    K_force, K_center, wl_values, alpha_values, psi_values = load_buoyancy_table(
        table_path, wl_values, alpha_values, psi_values, scale
    )
    na, npsi, nwl = K_force.shape[:3]
    K = np.empty((na, npsi, nwl), dtype=object)
    for i in range(na):
        for j in range(npsi):
            for k in range(nwl):
                K[i, j, k] = {"force": K_force[i, j, k], "center": K_center[i, j, k]}
    return K, wl_values, alpha_values, psi_values


load_table_to_3darray_Kirpich = load_table_to_3darray


def _lagrange_weights(nodes: np.ndarray, x: float) -> np.ndarray:
    nodes = np.asarray(nodes, dtype=np.float64)
    diff = x - nodes
    weights = np.empty(nodes.size, dtype=np.float64)
    for a in range(nodes.size):
        denom = nodes[a] - nodes
        denom[a] = 1.0
        terms = diff / denom
        terms[a] = 1.0
        weights[a] = np.prod(terms)
    return weights


def _quadratic_stencil(axis_values, x: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(axis_values, dtype=np.float64)
    n = values.size
    if n <= 2:
        return np.arange(n, dtype=int), _lagrange_weights(values, float(x))

    idx = int(np.searchsorted(values, x, side="left") - 1)
    idx = max(0, min(idx, n - 2))
    start = idx - 1
    if start < 0:
        start = 0
    elif start > n - 3:
        start = n - 3
    indices = np.arange(start, start + 3, dtype=int)
    return indices, _lagrange_weights(values[indices], float(x))


def _clamp_to_axis(axis_values, x: float) -> float:
    values = np.asarray(axis_values, dtype=np.float64)
    return float(min(max(float(x), values[0]), values[-1]))


def triquadratic_interpolation(
    wl,
    alpha,
    psi,
    K_force: np.ndarray,
    K_center: np.ndarray,
    wl_values,
    alpha_values,
    psi_values,
):
    """Квадратичная интерполяция. Вне сетки берётся крайнее значение."""
    wl = _clamp_to_axis(wl_values, wl)
    alpha = _clamp_to_axis(alpha_values, alpha)
    psi = _clamp_to_axis(psi_values, psi)

    ia, wa = _quadratic_stencil(alpha_values, alpha)
    ip, wp = _quadratic_stencil(psi_values, psi)
    iw, ww = _quadratic_stencil(wl_values, wl)

    weights = wa[:, None, None] * wp[None, :, None] * ww[None, None, :]
    force_block = K_force[np.ix_(ia, ip, iw)]
    center_block = K_center[np.ix_(ia, ip, iw)]

    interp_force = np.einsum("ijk,ijkl->l", weights, force_block)
    interp_center = np.einsum("ijk,ijkl->l", weights, center_block)

    if interp_force[2] > 225449.68297545:
        interp_force[2] = 225449.68297545
    if interp_force[2] < 1e-6:
        interp_force[2] = 0.0
    return interp_center, interp_force


def trilinear_interpolation(
    wl,
    alpha,
    psi,
    K_force: np.ndarray,
    K_center: np.ndarray,
    wl_values,
    alpha_values,
    psi_values,
):
    """Старое имя квадратичной интерполяции."""
    return triquadratic_interpolation(
        wl, alpha, psi, K_force, K_center, wl_values, alpha_values, psi_values,
    )


def lookup_buoyancy(
    z: float,
    alpha_deg: float,
    psi_deg: float,
    K_force,
    K_center,
    wl_values,
    alpha_values,
    psi_values,
    water_level: float = 0.0,
):
    return triquadratic_interpolation(
        water_level - z, alpha_deg, psi_deg,
        K_force, K_center, wl_values, alpha_values, psi_values,
    )


def calculate_initial_equilibrium(
    mass: float,
    K_force,
    K_center,
    wl_values,
    alpha_values,
    psi_values,
    g: float = 9.81,
    wl_min: float | None = None,
    wl_max: float | None = None,
    tolerance: float = 0.01,
):
    target_force = mass * g
    wl_min = float(wl_values[0]) if wl_min is None else wl_min
    wl_max = float(wl_values[-1]) if wl_max is None else wl_max

    def buoyancy_difference(wl):
        _, force = triquadratic_interpolation(
            wl, 0.0, 0.0, K_force, K_center, wl_values, alpha_values, psi_values
        )
        return force[2] - target_force

    while wl_max - wl_min > tolerance:
        mid = (wl_min + wl_max) / 2
        if buoyancy_difference(mid) > 0:
            wl_max = mid
        else:
            wl_min = mid

    best_wl = (wl_min + wl_max) / 2
    print(f"Уровень воды для равновесия: {best_wl:.3f} м")
    return best_wl
