"""Перестановка осей OBJ и сохранение под новым именем.

x' = z, y' = x, z' = y.
"""

from pathlib import Path

import numpy as np

SRC = Path("data/model.obj")
DST = Path("data/model_turned.obj")

# Столбцы — образы базисных векторов: e_x → e_y, e_y → e_z, e_z → e_x.
AXIS_MAP = np.array([
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
])


def map_xyz(xyz: np.ndarray) -> np.ndarray:
    return xyz @ AXIS_MAP.T


def reverse_face(line: str) -> str:
    parts = line.split()
    return " ".join([parts[0], *reversed(parts[1:])]) + "\n"


def rotate_obj(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open(encoding="utf-8") as inp, dst.open("w", encoding="utf-8") as out:
        for line in inp:
            kind = line.split(" ", 1)[0]
            if kind in ("v", "vn"):
                parts = line.split()
                xyz = map_xyz(np.array(parts[1:4], dtype=float))
                rest = " ".join(parts[4:])
                tail = f" {rest}" if rest else ""
                out.write(f"{kind} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}{tail}\n")
            elif kind == "f" and np.linalg.det(AXIS_MAP) < 0.0:
                out.write(reverse_face(line))
            else:
                out.write(line)


def main():
    root = Path(__file__).resolve().parents[1]
    src = root / SRC
    dst = root / DST
    rotate_obj(src, dst)
    print(f"{src.name} → {dst}")
    print("x' = z, y' = x, z' = y")


if __name__ == "__main__":
    main()
