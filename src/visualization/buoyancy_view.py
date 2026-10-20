import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from src.forces.buoyancy import BoatModel


def plot_buoyancy(boat: BoatModel, water_level: float = 0.0, show_normals: bool = False):
    """Красный — над водой, синий — под водой, зелёная стрелка — суммарная сила."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    triangles = [poly.vertices for poly in boat.polygons]
    forces = [np.linalg.norm(poly.buoyancy_force) for poly in boat.polygons]
    max_force = max(forces) if forces else 1.0

    colors = []
    for poly in boat.polygons:
        magnitude = np.linalg.norm(poly.buoyancy_force)
        if magnitude == 0:
            colors.append([1, 0, 0, 0.6])
        else:
            intensity = 0.3 + 0.7 * (np.log1p(magnitude) / np.log1p(max_force))
            colors.append([0, 0, 1, intensity])

    mesh = Poly3DCollection(
        triangles, facecolors=colors, edgecolor="k", linewidths=0.3, alpha=0.7
    )
    ax.add_collection3d(mesh)

    if show_normals:
        for poly in boat.polygons[:100]:
            ax.quiver(
                poly.center[0], poly.center[1], poly.center[2],
                poly.normal[0] * 0.1, poly.normal[1] * 0.1, poly.normal[2] * 0.1,
                color="gray", arrow_length_ratio=0.3,
            )

    total_force = boat.get_total_buoyancy_force()
    center = boat.get_buoyancy_center()
    force_norm = np.linalg.norm(total_force)

    if force_norm > 1e-6:
        scale = 0.05 / force_norm
        ax.quiver(
            center[0], center[1], center[2],
            total_force[0] * scale, total_force[1] * scale, total_force[2] * scale,
            color="green", linewidth=2, arrow_length_ratio=0.2,
        )
        ax.scatter(center[0], center[1], center[2], color="orange", s=80, zorder=5)

    all_points = np.vstack(triangles)
    min_coord = np.min(all_points, axis=0)
    max_coord = np.max(all_points, axis=0)
    margin = 0.1 * (max_coord - min_coord + 1e-6)

    ax.set_xlim(min_coord[0] - margin[0], max_coord[0] + margin[0])
    ax.set_ylim(min_coord[1] - margin[1], max_coord[1] + margin[1])
    ax.set_zlim(min_coord[2] - margin[2], max_coord[2] + margin[2])
    ax.set_box_aspect(max_coord - min_coord + 2 * margin)
    ax.view_init(elev=18, azim=0)

    x_range = np.linspace(min_coord[0], max_coord[0], 2)
    y_range = np.linspace(min_coord[1], max_coord[1], 2)
    xx, yy = np.meshgrid(x_range, y_range)
    zz = np.full_like(xx, water_level)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color="cyan")

    ax.set_title(
        f"Сила Архимеда: {force_norm:.1f} N\n"
        f"Точка приложения: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})"
    )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.tight_layout()
    return fig, ax
