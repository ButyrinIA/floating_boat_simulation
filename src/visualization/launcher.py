"""Окно выбора сценария и правки config/simulation.json."""

import copy
import tkinter as tk
from tkinter import ttk

from src.utils.config import apply_overrides

MODES = (
    ("simulation", "Симуляция"),
    ("precompute", "Предрасчёт"),
    ("mesh", "Сетка модели"),
)

COLUMNS = (
    (
        "Общие",
        (
            ("mass", "Масса, кг", "float"),
            ("water_level", "Уровень воды, м", "float"),
            ("fluid_density", "Плотность, кг/м³", "float"),
            ("gravity", "g, м/с²", "float"),
            ("table_scale", "Масштаб таблицы", "float"),
            ("model_path", "Файл модели", "str"),
            ("buoyancy_table_path", "Файл таблицы", "str"),
        ),
    ),
    (
        "Симуляция",
        (
            ("simulation.t_max", "t_max, с", "float"),
            ("simulation.dt", "dt, с", "float"),
            ("simulation.initial_z", "Начальное Z, м", "float"),
            ("simulation.animation_stride", "Шаг анимации", "int"),
            ("simulation.trust_force_body", "Тяга Fx, Fy, Fz", "csv"),
            ("simulation.initial_quaternion", "Кватернион q0..q3", "csv"),
        ),
    ),
    (
        "Предрасчёт и сетка",
        (
            ("precompute_grid.water_level.min", "Осадка min, м", "float"),
            ("precompute_grid.water_level.max", "Осадка max, м", "float"),
            ("precompute_grid.water_level.step", "Осадка шаг, м", "float"),
            ("precompute_grid.alpha_deg.min", "Крен min, °", "float"),
            ("precompute_grid.alpha_deg.max", "Крен max, °", "float"),
            ("precompute_grid.alpha_deg.step", "Крен шаг, °", "float"),
            ("precompute_grid.psi_deg.min", "Дифферент min, °", "float"),
            ("precompute_grid.psi_deg.max", "Дифферент max, °", "float"),
            ("precompute_grid.psi_deg.step", "Дифферент шаг, °", "float"),
            ("mesh_subdivide", "Субдивизии сетки", "int"),
        ),
    ),
)


def _dig(config: dict, key: str):
    node = config
    for part in key.split("."):
        node = node[part]
    return node


def _initial_text(config: dict, key: str, kind: str) -> str:
    if key == "mesh_subdivide":
        value = config.get("mesh_subdivide", 0)
    else:
        value = _dig(config, key)
    if kind == "csv":
        return ", ".join(f"{v:g}" if isinstance(v, float) else str(v) for v in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _parse_value(text: str, kind: str, label: str):
    raw = text.strip().replace(",", ".") if kind != "csv" else text.strip()
    if kind == "str":
        if not raw:
            raise ValueError(f"{label}: пустое значение")
        return raw
    if kind == "int":
        try:
            return int(float(raw.replace(",", ".")))
        except ValueError:
            raise ValueError(f"{label}: нужно целое число") from None
    if kind == "float":
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            raise ValueError(f"{label}: нужно число") from None
    parts = [p.strip() for p in text.replace(";", ",").split(",") if p.strip()]
    try:
        return [float(p.replace(",", ".")) for p in parts]
    except ValueError:
        raise ValueError(f"{label}: числа через запятую") from None


def read_form(config: dict, values: dict[str, str], equilibrium: bool, animate: bool) -> dict:
    overrides = {}
    for _title, fields in COLUMNS:
        for key, label, kind in fields:
            overrides[key] = _parse_value(values[key], kind, label)
    overrides["simulation.use_equilibrium_z"] = bool(equilibrium)
    overrides["simulation.animate"] = bool(animate)
    updated = copy.deepcopy(config)
    apply_overrides(updated, overrides)
    return updated


def serve_launcher(config: dict, initial_mode: str, on_choice) -> None:
    """Прячет окно на время сценария и показывает его снова после on_choice."""
    mode_ids = [mode for mode, _label in MODES]
    if initial_mode not in mode_ids:
        initial_mode = mode_ids[0]

    root = tk.Tk()
    root.title("Лодка")
    root.minsize(920, 560)

    left = ttk.Frame(root, padding=16)
    left.grid(row=0, column=0, sticky="nsw")
    ttk.Label(left, text="Сценарий", font=("", 13, "bold")).pack(anchor="w", pady=(0, 8))

    mode_var = tk.StringVar(value=initial_mode)
    for mode, label in MODES:
        ttk.Radiobutton(left, text=label, value=mode, variable=mode_var).pack(anchor="w", pady=3)

    equilibrium = tk.BooleanVar(
        value=bool(config.get("simulation", {}).get("use_equilibrium_z", True))
    )
    animate = tk.BooleanVar(
        value=bool(config.get("simulation", {}).get("animate", True))
    )
    ttk.Separator(left).pack(fill="x", pady=14)
    ttk.Checkbutton(left, text="Равновесное Z", variable=equilibrium).pack(anchor="w", pady=4)
    ttk.Checkbutton(left, text="Анимация", variable=animate).pack(anchor="w", pady=4)

    form = ttk.Frame(root, padding=(8, 16, 16, 8))
    form.grid(row=0, column=1, sticky="nsew")
    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)

    entries: dict[str, tk.StringVar] = {}
    for col, (title, fields) in enumerate(COLUMNS):
        box = ttk.LabelFrame(form, text=title, padding=8)
        box.grid(row=0, column=col, sticky="nsew", padx=6)
        form.columnconfigure(col, weight=1)
        for i, (key, label, kind) in enumerate(fields):
            ttk.Label(box, text=label).grid(row=i, column=0, sticky="w", padx=(0, 8), pady=3)
            var = tk.StringVar(value=_initial_text(config, key, kind))
            ttk.Entry(box, textvariable=var, width=24).grid(row=i, column=1, sticky="ew", pady=3)
            entries[key] = var
        box.columnconfigure(1, weight=1)

    status = ttk.Label(
        root,
        text="Одно окно параметров. На время сценария оно скрывается и возвращается, когда закроете окно сценария.",
        wraplength=720,
    )
    status.grid(row=1, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))

    state: dict = {"result": None}

    def on_run():
        try:
            updated = read_form(
                config,
                {key: var.get() for key, var in entries.items()},
                equilibrium.get(),
                animate.get(),
            )
        except ValueError as exc:
            status.configure(text=str(exc), foreground="red")
            return
        updated["mode"] = mode_var.get()
        state["result"] = updated
        root.quit()

    def on_close():
        state["result"] = None
        root.quit()

    ttk.Button(left, text="Запустить", command=on_run).pack(anchor="w", pady=(18, 0))
    root.protocol("WM_DELETE_WINDOW", on_close)

    try:
        while True:
            state["result"] = None
            root.deiconify()
            root.lift()
            root.mainloop()
            choice = state["result"]
            if choice is None:
                return
            root.withdraw()
            root.update()
            on_choice(choice)
            if not root.winfo_exists():
                return
    finally:
        if root.winfo_exists():
            root.destroy()
