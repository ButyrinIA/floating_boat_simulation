import itertools
import pickle
from pathlib import Path

import numpy as np
import trimesh
from tqdm import tqdm


class Polygon3D:
    """Треугольный полигон: центр, площадь, нормаль, сила Архимеда."""

    def __init__(self, vertices: np.ndarray, normal: np.ndarray):
        self.vertices = vertices
        self.normal = normal / np.linalg.norm(normal)
        self.center = np.mean(vertices, axis=0)

        v1 = vertices[1] - vertices[0]
        v2 = vertices[2] - vertices[0]
        self.area = 0.5 * np.linalg.norm(np.cross(v1, v2))
        self.buoyancy_force = np.zeros(3)

    def compute_buoyancy_force(
        self,
        water_level: float = 0.0,
        rho: float = 1000.0,
        g: float = 9.81,
    ) -> np.ndarray:
        """
        Сила Архимеда на полигон: P = rho * g * h, F = P * area.
        h — глубина центра полигона под уровнем воды (water_level - z).
        """
        depth = water_level - self.center[2]

        if depth > 0:
            pressure = rho * g * depth
            magnitude = pressure * self.area
            # нормаль вниз (n_z < 0) даёт выталкивающую силу вверх
            self.buoyancy_force = np.array([0.0, 0.0, -magnitude * self.normal[2]])
        else:
            self.buoyancy_force = np.zeros(3)

        return self.buoyancy_force


class BoatModel:
    """3D-модель корпуса: триангуляция, расчёт и суммирование сил Архимеда."""

    def __init__(self, obj_file: str | Path):
        self.mesh = self._load_mesh(obj_file)
        self.polygons = self._create_polygons()

    @staticmethod
    def _load_mesh(file_path: str | Path) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(file_path)
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError("Модель должна быть триангулированной сеткой")
        return mesh

    def _create_polygons(self) -> list[Polygon3D]:
        return [
            Polygon3D(self.mesh.triangles[i], self.mesh.face_normals[i])
            for i in range(len(self.mesh.triangles))
        ]

    def compute_buoyancy(self, water_level: float = 0.0, rho: float = 1000.0, g: float = 9.81):
        for poly in self.polygons:
            poly.compute_buoyancy_force(water_level, rho, g)

    def get_total_buoyancy_force(self) -> np.ndarray:
        total = np.zeros(3)
        for poly in self.polygons:
            total += poly.buoyancy_force
        return total

    def get_buoyancy_center(self) -> np.ndarray:
        """Точка приложения — центр масс сил по модулям на полигонах."""
        total_force = self.get_total_buoyancy_force()
        if np.linalg.norm(total_force) < 1e-6:
            return np.zeros(3)

        center = np.zeros(3)
        total_magnitude = 0.0
        for poly in self.polygons:
            force_magnitude = np.linalg.norm(poly.buoyancy_force)
            center += poly.center * force_magnitude
            total_magnitude += force_magnitude

        return center / total_magnitude

    def rotate_boat(self, alpha: float = 0.0, psi: float = 0.0, phi: float = 0.0):
        """Вращение вокруг осей X (alpha), Y (psi), Z (phi), в радианах."""
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(alpha), -np.sin(alpha)],
            [0, np.sin(alpha), np.cos(alpha)],
        ])
        Ry = np.array([
            [np.cos(psi), 0, np.sin(psi)],
            [0, 1, 0],
            [-np.sin(psi), 0, np.cos(psi)],
        ])
        Rz = np.array([
            [np.cos(phi), -np.sin(phi), 0],
            [np.sin(phi), np.cos(phi), 0],
            [0, 0, 1],
        ])
        rotation = Rx @ Ry @ Rz

        self.mesh.vertices = self.mesh.vertices @ rotation.T
        for poly in self.polygons:
            poly.vertices = poly.vertices @ rotation.T
            poly.normal = poly.normal @ rotation.T
            poly.center = np.mean(poly.vertices, axis=0)

    def refine_mesh(self, subdivisions: int = 1):
        self.mesh = self.mesh.subdivide(subdivisions)
        self.polygons = self._create_polygons()


def precompute_buoyancy_table(
    obj_file: str | Path,
    water_levels: np.ndarray,
    alpha_values: np.ndarray,
    psi_values: np.ndarray,
    save_path: str | Path,
):
    """Таблица силы и центра давления по осадке, крену и дифференту."""
    table = {}
    alpha_rad = np.radians(alpha_values)
    psi_rad = np.radians(psi_values)
    water_levels = np.round(water_levels, 3)

    positive_levels = water_levels[water_levels >= 0]
    negative_levels = water_levels[water_levels < 0]

    for alpha, psi in tqdm(
        itertools.product(alpha_rad, psi_rad),
        desc="Расчёт углов",
        total=len(alpha_rad) * len(psi_rad),
    ):
        boat = BoatModel(obj_file)
        boat.rotate_boat(alpha=alpha, psi=psi, phi=0)

        max_force = None
        max_center = None
        reached_max = False

        for wl in positive_levels:
            if not reached_max:
                boat.compute_buoyancy(water_level=wl)
                force = boat.get_total_buoyancy_force()
                center = boat.get_buoyancy_center()
                if max_force is None or np.linalg.norm(force) > np.linalg.norm(max_force):
                    max_force = force
                    max_center = center
                else:
                    reached_max = True

            key = (wl, round(np.degrees(alpha), 1), round(np.degrees(psi), 1))
            table[key] = {"force": max_force, "center": max_center}

        zero_force_found = False
        zero_force_center = np.zeros(3)

        for wl in negative_levels:
            if not zero_force_found:
                boat.compute_buoyancy(water_level=wl)
                force = boat.get_total_buoyancy_force()
                center = boat.get_buoyancy_center()
                if np.linalg.norm(force) < 1e-6:
                    zero_force_found = True
                    zero_force_center = center
                    force = np.zeros(3)
            else:
                force = np.zeros(3)
                center = zero_force_center

            key = (wl, round(np.degrees(alpha), 1), round(np.degrees(psi), 1))
            table[key] = {"force": force, "center": center}

    from src.utils.table_loader import save_table_npz, table_dict_to_arrays

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    K_force, K_center = table_dict_to_arrays(table, water_levels, alpha_values, psi_values, scale=1.0)
    npz_path = save_path.with_suffix(".npz")
    save_table_npz(K_force, K_center, water_levels, alpha_values, psi_values, npz_path)

    with open(save_path, "wb") as f:
        pickle.dump(table, f)

    return table
