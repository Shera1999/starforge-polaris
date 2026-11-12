"""Small, self-contained POLARIS uniform-octree writer."""

from __future__ import annotations

import struct


class Cell:
    def __init__(self, x: float, y: float, z: float, length: float, level: int):
        self.x = x
        self.y = y
        self.z = z
        self.length = length
        self.level = level
        self.is_leaf = False
        self.data = None
        self.children = None


class Octree:
    """Uniform-depth octree using POLARIS grid format ID 20."""

    def __init__(self, side: float):
        self.root = Cell(-side / 2, -side / 2, -side / 2, side, 0)
        self.leaf_count = 0

    @staticmethod
    def _split(cell: Cell) -> None:
        half = cell.length / 2
        cell.children = [
            Cell(
                cell.x + (index & 1) * half,
                cell.y + ((index >> 1) & 1) * half,
                cell.z + ((index >> 2) & 1) * half,
                half,
                cell.level + 1,
            )
            for index in range(8)
        ]

    def insert(self, xyz, level: int, values) -> None:
        cell = self.root
        while cell.level < level:
            if cell.children is None:
                self._split(cell)
            index = (
                (xyz[0] >= cell.x + cell.length / 2)
                | ((xyz[1] >= cell.y + cell.length / 2) << 1)
                | ((xyz[2] >= cell.z + cell.length / 2) << 2)
            )
            cell = cell.children[int(index)]
        if cell.is_leaf:
            raise ValueError("Two samples map to the same octree leaf")
        cell.is_leaf = True
        cell.data = list(values)
        self.leaf_count += 1

    def validate(self, cell=None) -> bool:
        cell = self.root if cell is None else cell
        if cell.is_leaf:
            return cell.data is not None and cell.children is None
        return (
            cell.children is not None
            and len(cell.children) == 8
            and all(self.validate(child) for child in cell.children)
        )

    @staticmethod
    def _write_cell(handle, cell: Cell) -> None:
        handle.write(struct.pack("<HH", int(cell.is_leaf), cell.level))
        if cell.is_leaf:
            handle.write(struct.pack(f"<{len(cell.data)}f", *cell.data))
        else:
            for child in cell.children:
                Octree._write_cell(handle, child)

    def write(self, path, field_ids) -> None:
        if not self.validate():
            raise RuntimeError("Refusing to write an incomplete octree")
        with open(path, "wb") as handle:
            handle.write(struct.pack("<HH", 20, len(field_ids)))
            handle.write(struct.pack(f"<{len(field_ids)}H", *field_ids))
            handle.write(struct.pack("<d", self.root.length))
            self._write_cell(handle, self.root)
