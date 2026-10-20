"""Уравнения движения лодки: Архимед из таблицы, сопротивление, тяга."""

from dataclasses import dataclass

import numpy as np
from scipy.special import ellipeinc, ellipkinc

from src.utils.rotations import Rot3D, TRot3D
from src.utils.table_loader import triquadratic_interpolation


def ellipsoid_surface_area(rx: float, ry: float, rz: float) -> float:
    """Площадь сплошного трёхосного эллипсоида с полуосями rx, ry, rz."""
    a, b, c = sorted((abs(rx), abs(ry), abs(rz)), reverse=True)
    if a <= 0.0:
        return 0.0
    if (a - c) / a < 1e-9:
        return float(4.0 * np.pi * a * a)
    phi = float(np.arccos(np.clip(c / a, -1.0, 1.0)))
    m = (a * a * (b * b - c * c)) / (b * b * (a * a - c * c))
    m = float(np.clip(m, 0.0, 1.0))
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return float(
        2.0 * np.pi * c * c
        + (2.0 * np.pi * a * b / sin_phi)
        * (ellipeinc(phi, m) * sin_phi**2 + ellipkinc(phi, m) * cos_phi**2)
    )


@dataclass
class BoatParams:
    mass: float = 7000.0
    rho: float = 1000.0
    g: float = 9.81

    R_x: float = 2.238104 / 2
    R_y: float = 10.578384 / 2
    R_z: float = 1.806256 / 2

    # 0 — посчитать площадь эллипсоида по полуосям R_x, R_y, R_z.
    wetted_area: float = 0.0
    length: float = 10.578384
    nu: float = 1.0e-6
    # 1+k: линия ITTC на 53.432 м² при Re=1.1e7 даёт 114 Н (Ventorah, сплошной корпус, 1 м/с).
    form_factor_1k: float = 114.0 / (
        0.5 * 1000.0 * 53.432 * (0.075 / (np.log10(1.1e7) - 2.0) ** 2)
    )
    Cd_cross: float = 0.8
    roughness_m: float = 150e-6
    Cvx: float = 6.0
    Cvy: float = 6.0
    Cvz: float = 3.0

    trust_body: tuple[float, float, float] = (0.0, 20000.0, 0.0)
    trust_arm_body: tuple[float, float, float] = (0.0, -10.578384 / 2, -1.806256 / 2)

    def __post_init__(self):
        if self.wetted_area <= 0.0:
            self.wetted_area = ellipsoid_surface_area(self.R_x, self.R_y, self.R_z)
        if self.length <= 0.0:
            self.length = 2.0 * max(self.R_x, self.R_y, self.R_z)

    @property
    def inertia(self) -> tuple[float, float, float]:
        m = self.mass
        A = (m / 5) * (self.R_y**2 + self.R_z**2)
        B = (m / 5) * (self.R_x**2 + self.R_z**2)
        C = (m / 5) * (self.R_y**2 + self.R_x**2)
        return A, B, C

    @property
    def cross_sections(self) -> tuple[float, float, float]:
        Sx = np.pi * self.R_z * self.R_y
        Sy = np.pi * self.R_z * self.R_x
        Sz = np.pi * self.R_x * self.R_y
        return Sx, Sy, Sz


@dataclass
class BuoyancyTableContext:
    K_force: np.ndarray
    K_center: np.ndarray
    wl_values: np.ndarray
    alpha_values: np.ndarray
    psi_values: np.ndarray
    water_level: float = 0.0


def _angles_from_quaternion(q0, q1, q2, q3) -> tuple[float, float]:
    _, _, sin_alpha = Rot3D(0, 1, 0, q0, q1, q2, q3)
    _, _, m_cos_alpha_sin_psi = Rot3D(1, 0, 0, q0, q1, q2, q3)
    alpha = np.rad2deg(np.arcsin(np.clip(sin_alpha, -1.0, 1.0)))
    psi = np.rad2deg(np.arcsin(np.clip(-m_cos_alpha_sin_psi, -1.0, 1.0)))
    return alpha, psi


def skin_friction_coefficient(reynolds: float) -> float:
    """Линия ITTC-57. Ниже Re=100 формула не определена."""
    if reynolds <= 100.0:
        return 0.0
    return 0.075 / (np.log10(reynolds) - 2.0) ** 2


def roughness_allowance(reynolds: float, length: float, ks: float) -> float:
    """Надбавка ITTC на шероховатость, прибавляется к Cf."""
    if reynolds <= 100.0 or length <= 0.0 or ks <= 0.0:
        return 0.0
    allowance = 0.044 * ((ks / length) ** (1.0 / 3.0) - 10.0 * reynolds ** (-1.0 / 3.0)) + 0.000125
    return float(max(0.0, allowance))


def _volume_fraction(zeta: float) -> float:
    z = float(np.clip(zeta, -1.0, 1.0))
    return (2.0 + 3.0 * z - z ** 3) / 4.0


def _disk_fraction(zeta: float) -> float:
    z = float(np.clip(zeta, -1.0, 1.0))
    return float((np.arccos(-z) + z * np.sqrt(max(0.0, 1.0 - z * z))) / np.pi)


def _horizontal_axes(params: BoatParams) -> tuple[int, int]:
    """Длинная и короткая оси ватерлинии. Вертикаль корпуса — Z."""
    radii = (params.R_x, params.R_y)
    i_long = int(np.argmax(radii))
    i_lat = 1 - i_long
    return i_long, i_lat


def _wave_resistance(
    speed: float,
    r_long: float,
    r_beam: float,
    r_vert: float,
    zeta: float,
    params: BoatParams,
) -> float:
    """Волновое сопротивление Холтропа–Меннена, Н. Нет бульба и транца."""
    if zeta <= -1.0 or zeta >= 1.0 or speed <= 0.0:
        return 0.0
    span = float(np.sqrt(max(0.0, 1.0 - zeta * zeta)))
    length = 2.0 * r_long * span
    beam = 2.0 * r_beam * span
    draft = r_vert * (1.0 + zeta)
    if length < 0.5 or beam < 0.05 or draft < 0.02:
        return 0.0
    volume = _volume_fraction(zeta) * (4.0 / 3.0 * np.pi * r_long * r_beam * r_vert)
    midship = _disk_fraction(zeta) * np.pi * r_beam * r_vert
    if volume <= 0.0 or midship <= 0.0:
        return 0.0
    fn = speed / np.sqrt(params.g * length)
    if fn < 0.05:
        return 0.0
    cp = float(np.clip(volume / (midship * length), 0.2, 0.85))
    cwp = 0.25 * np.pi
    lcb = 0.0
    length_run = length * (1.0 - cp)
    ratio_lb = length / beam
    arg = (
        ratio_lb ** 0.80856
        * (1.0 - cwp) ** 0.30484
        * max(1.0 - cp - 0.0225 * lcb, 1e-3) ** 0.6367
        * max(length_run / beam, 1e-3) ** 0.34574
        * max(100.0 * volume / length ** 3, 1e-6) ** 0.16302
    )
    entrance = 1.0 + 89.0 * np.exp(-arg)
    bl = beam / length
    if bl < 0.11:
        c7 = 0.229577 * bl ** (1.0 / 3.0)
    elif bl < 0.25:
        c7 = bl
    else:
        c7 = 0.5 - 0.0625 * ratio_lb
    c1 = (
        2223105.0
        * c7 ** 3.78613
        * (draft / beam) ** 1.07961
        * max(90.0 - entrance, 1.0) ** (-1.37565)
    )
    lam = 1.446 * cp - (0.03 * ratio_lb if ratio_lb < 12.0 else 0.36)
    if cp < 0.80:
        c16 = 8.07981 * cp - 13.8673 * cp ** 2 + 6.984388 * cp ** 3
    else:
        c16 = 1.73014 - 0.7067 * cp
    slenderness = length ** 3 / volume
    if slenderness < 512.0:
        c15 = -1.69385
    elif slenderness < 1726.91:
        c15 = -1.69385 + (length / volume ** (1.0 / 3.0) - 8.0) / 2.36
    else:
        c15 = 0.0
    m1 = (
        0.0140407 * length / draft
        - 1.75254 * volume ** (1.0 / 3.0) / length
        - 4.79323 * bl
        - c16
    )
    m2 = c15 * cp ** 2 * np.exp(-0.1 * fn ** (-2))
    exponent = m1 * fn ** (-0.9) + m2 * np.cos(lam * fn ** (-2))
    exponent = float(np.clip(exponent, -40.0, 2.0))
    resistance = c1 * volume * params.rho * params.g * np.exp(exponent)
    displacement = volume * params.rho * params.g
    return float(np.clip(resistance, 0.0, 0.25 * displacement))


def drag_force_body(
    vx: float,
    vy: float,
    vz: float,
    immersion: float,
    params: BoatParams,
) -> tuple[float, float, float]:
    """Сопротивление на плаву: трение вдоль длины, волна и поперечный обдув.

    immersion — высота воды над центром эллипсоида, м.
    Компоненты имеют знак скорости: в уравнениях движения они вычитаются.
    """
    radii = (params.R_x, params.R_y, params.R_z)
    r_vert = radii[2]
    if r_vert <= 0.0:
        return 0.0, 0.0, 0.0
    zeta = immersion / r_vert
    if zeta <= -1.0:
        return 0.0, 0.0, 0.0
    i_long, i_lat = _horizontal_axes(params)
    r_long = radii[i_long]
    r_beam = radii[i_lat]
    wet = 1.0 if zeta >= 1.0 else 0.5 * (zeta + 1.0)
    area = wet * params.wetted_area
    velocity = (vx, vy, vz)
    force = [0.0, 0.0, 0.0]
    ahead = velocity[i_long]
    if abs(ahead) > 1e-12 and area > 0.0 and r_long > 0.0:
        speed = abs(ahead)
        length = 2.0 * r_long * (1.0 if zeta >= 1.0 else np.sqrt(max(0.0, 1.0 - min(zeta, 1.0) ** 2)))
        reynolds = speed * length / params.nu
        cf = skin_friction_coefficient(reynolds) + roughness_allowance(reynolds, length, params.roughness_m)
        viscous = 0.5 * params.rho * speed * speed * area * cf * params.form_factor_1k
        wave = _wave_resistance(speed, r_long, r_beam, r_vert, zeta, params)
        force[i_long] = (viscous + wave) * np.sign(ahead)
    side = velocity[i_lat]
    if abs(side) > 1e-12:
        silhouette = np.pi * r_long * r_vert * (1.0 if zeta >= 1.0 else _disk_fraction(zeta))
        force[i_lat] = 0.5 * params.rho * params.Cd_cross * silhouette * side * abs(side)
    heave = velocity[2]
    if abs(heave) > 1e-12:
        if zeta >= 1.0:
            plan = np.pi * r_long * r_beam
        else:
            plan = np.pi * r_long * r_beam * max(0.0, 1.0 - zeta * zeta)
        force[2] = 0.5 * params.rho * params.Cd_cross * plan * heave * abs(heave)
    return force[0], force[1], force[2]


def state_derivative(
    state: np.ndarray,
    params: BoatParams,
    buoyancy: BuoyancyTableContext,
) -> np.ndarray:
    X, Y, Z, q0, q1, q2, q3, VX, VY, VZ, Omegax, Omegay, Omegaz = state
    m = params.mass
    rho = params.rho
    g = params.g
    A, B, C = params.inertia
    Sx, _, Sz = params.cross_sections

    Vx, Vy, Vz = TRot3D(VX, VY, VZ, q0, q1, q2, q3)

    alpha_deg, psi_deg = _angles_from_quaternion(q0, q1, q2, q3)
    immersion = buoyancy.water_level - Z
    P_A, F_A = triquadratic_interpolation(
        immersion,
        alpha_deg,
        psi_deg,
        buoyancy.K_force,
        buoyancy.K_center,
        buoyancy.wl_values,
        buoyancy.alpha_values,
        buoyancy.psi_values,
    )

    M_A = np.cross(P_A, F_A)
    M_Ax, M_Ay, M_Az = TRot3D(M_A[0], M_A[1], M_A[2], q0, q1, q2, q3)

    FSx, FSy, FSz = drag_force_body(Vx, Vy, Vz, immersion, params)
    FSX, FSY, FSZ = Rot3D(FSx, FSy, FSz, q0, q1, q2, q3)

    F_trust_x, F_trust_y, F_trust_z = params.trust_body
    F_trust_X, F_trust_Y, F_trust_Z = Rot3D(
        F_trust_x, F_trust_y, F_trust_z, q0, q1, q2, q3
    )
    M_trust = np.cross(
        np.array(params.trust_arm_body),
        np.array(params.trust_body),
    )

    MSx = rho * params.Cvx * Sz * Omegax * abs(Omegax) / 2
    MSy = rho * params.Cvy * Sz * Omegay * abs(Omegay) / 2
    MSz = rho * params.Cvz * Sx * Omegaz * abs(Omegaz) / 2

    FX = F_A[0] - FSX + F_trust_X
    FY = F_A[1] - FSY + F_trust_Y
    FZ = F_A[2] - m * g - FSZ + F_trust_Z

    MCx = M_Ax - MSx + M_trust[0]
    MCy = M_Ay - MSy + M_trust[1]
    MCz = M_Az - MSz + M_trust[2]

    dX, dY, dZ = VX, VY, VZ
    dq0 = 0.5 * (-q1 * Omegax - q2 * Omegay - q3 * Omegaz)
    dq1 = 0.5 * (q0 * Omegax + q2 * Omegaz - q3 * Omegay)
    dq2 = 0.5 * (q0 * Omegay - q1 * Omegaz + q3 * Omegax)
    dq3 = 0.5 * (q0 * Omegaz + q1 * Omegay - q2 * Omegax)

    dVX, dVY, dVZ = FX / m, FY / m, FZ / m

    dOmega_x = (MCx - (C - B) * Omegay * Omegaz) / A
    dOmega_y = (MCy - (A - C) * Omegaz * Omegax) / B
    dOmega_z = (MCz - (B - A) * Omegax * Omegay) / C

    return np.array([
        dX, dY, dZ, dq0, dq1, dq2, dq3,
        dVX, dVY, dVZ, dOmega_x, dOmega_y, dOmega_z,
    ])


def normalize_quaternion_in_state(state: np.ndarray) -> np.ndarray:
    state = state.copy()
    q = np.asarray(state[3:7], dtype=float)
    norm = np.linalg.norm(q)
    if np.isfinite(norm) and norm > 1e-12:
        state[3:7] = q / norm
    else:
        # 0,0,0,0 не задаёт ориентацию; единичный кватернион — лодка без крена
        state[3:7] = (1.0, 0.0, 0.0, 0.0)
    return state
