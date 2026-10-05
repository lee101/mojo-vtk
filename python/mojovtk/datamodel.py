"""Data model mirroring `vtkCommonDataModel`.

`vtkPolyData` and `vtkUnstructuredGrid` are pipelines in VTK; here they are
plain containers over the exact arrays VTK stores, so a caller can move between
the two libraries by swapping the container and nothing else.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64

VTK_EMPTY_CELL = 0
VTK_VERTEX = 1
VTK_POLY_VERTEX = 2
VTK_LINE = 3
VTK_POLY_LINE = 4
VTK_TRIANGLE = 5
VTK_TRIANGLE_STRIP = 6
VTK_POLYGON = 7
VTK_PIXEL = 8
VTK_QUAD = 9
VTK_TETRA = 10
VTK_VOXEL = 11
VTK_HEXAHEDRON = 12
VTK_WEDGE = 13
VTK_PYRAMID = 14
VTK_PENTAGONAL_PRISM = 15
VTK_HEXAGONAL_PRISM = 16
VTK_INVALID = 1000

# vtkCellType.GetClassNameFromTypeId
_CELL_TYPE_NAMES = {
    VTK_EMPTY_CELL: "VTK_EMPTY_CELL",
    VTK_VERTEX: "VTK_VERTEX",
    VTK_POLY_VERTEX: "VTK_POLY_VERTEX",
    VTK_LINE: "VTK_LINE",
    VTK_POLY_LINE: "VTK_POLY_LINE",
    VTK_TRIANGLE: "VTK_TRIANGLE",
    VTK_TRIANGLE_STRIP: "VTK_TRIANGLE_STRIP",
    VTK_POLYGON: "VTK_POLYGON",
    VTK_PIXEL: "VTK_PIXEL",
    VTK_QUAD: "VTK_QUAD",
    VTK_TETRA: "VTK_TETRA",
    VTK_VOXEL: "VTK_VOXEL",
    VTK_HEXAHEDRON: "VTK_HEXAHEDRON",
    VTK_WEDGE: "VTK_WEDGE",
    VTK_PYRAMID: "VTK_PYRAMID",
    VTK_PENTAGONAL_PRISM: "VTK_PENTAGONAL_PRISM",
    VTK_HEXAGONAL_PRISM: "VTK_HEXAGONAL_PRISM",
    VTK_INVALID: "VTK_INVALID",
}

# vtkCellTypeUtilities::GetDimension
_CELL_DIMENSIONS = {
    VTK_EMPTY_CELL: -1,
    VTK_VERTEX: 0,
    VTK_POLY_VERTEX: 0,
    VTK_LINE: 1,
    VTK_POLY_LINE: 1,
    VTK_TRIANGLE: 2,
    VTK_TRIANGLE_STRIP: 2,
    VTK_POLYGON: 2,
    VTK_PIXEL: 2,
    VTK_QUAD: 2,
    VTK_TETRA: 3,
    VTK_VOXEL: 3,
    VTK_HEXAHEDRON: 3,
    VTK_WEDGE: 3,
    VTK_PYRAMID: 3,
    VTK_PENTAGONAL_PRISM: 3,
    VTK_HEXAGONAL_PRISM: 3,
}


def GetClassNameFromTypeId(cellType: int) -> str:
    """`vtkCellType.GetClassNameFromTypeId`."""
    return _CELL_TYPE_NAMES.get(int(cellType), "")


def GetDimension(cellType: int) -> int:
    """`vtkCellTypeUtilities::GetDimension`."""
    return _CELL_DIMENSIONS.get(int(cellType), -1)


def poly_cell_type(npts: int) -> int:
    """`vtkPolyData::GetCellType` for a poly of `npts` points."""
    if npts == 3:
        return VTK_TRIANGLE
    if npts == 4:
        return VTK_QUAD
    return VTK_POLYGON


def empty_cell_array() -> tuple[np.ndarray, np.ndarray]:
    return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)


def as_cell_array(cells) -> tuple[np.ndarray, np.ndarray]:
    """Accept either a (connectivity, offsets) pair or a VTK-style nested list."""
    if isinstance(cells, CellArray):
        return cells.connectivity, cells.offsets
    if cells is None:
        return empty_cell_array()
    if isinstance(cells, tuple) and len(cells) == 2:
        conn, off = cells
        return i64(np.asarray(conn, dtype=np.int64).ravel()), i64(
            np.asarray(off, dtype=np.int64).ravel()
        )
    flat: list[int] = []
    offsets = [0]
    for cell in cells:
        for v in np.asarray(cell).ravel().tolist():
            flat.append(int(v))
        offsets.append(len(flat))
    return (
        np.asarray(flat, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


class CellArray:
    """`vtkCellArray` in its modern (connectivity, offsets) layout."""

    __slots__ = ("connectivity", "offsets")

    def __init__(self, cells=()):
        self.connectivity, self.offsets = as_cell_array(cells)

    def GetNumberOfCells(self) -> int:
        return int(self.offsets.size) - 1

    def GetNumberOfIds(self, cell: int) -> int:
        return int(self.offsets[cell + 1] - self.offsets[cell])

    def GetId(self, cell: int, i: int) -> int:
        return int(self.connectivity[int(self.offsets[cell]) + i])

    def GetCell(self, cell: int) -> np.ndarray:
        return self.connectivity[
            int(self.offsets[cell]) : int(self.offsets[cell + 1])
        ]

    def GetCells(self) -> list[np.ndarray]:
        return [self.GetCell(c) for c in range(self.GetNumberOfCells())]

    def max_cell_size(self) -> int:
        if self.GetNumberOfCells() == 0:
            return 0
        d = np.diff(self.offsets.astype(np.int64))
        return int(d.max()) if d.size else 0

    def total_ids(self) -> int:
        return int(self.offsets[-1]) if self.offsets.size else 0

    def cell_types(self) -> np.ndarray:
        """Cell types implied by the id counts, as vtkPolyData reports them."""
        d = np.diff(self.offsets.astype(np.int64))
        out = np.empty(d.size, dtype=np.int64)
        for i, n in enumerate(d.tolist()):
            out[i] = poly_cell_type(n)
        return out

    def take(self, src: np.ndarray) -> "CellArray":
        """Gather cells by index, keeping order (used for the outSrc map)."""
        offsets = [0]
        flat: list[int] = []
        for cell in np.asarray(src, dtype=np.int64).ravel().tolist():
            for v in self.GetCell(int(cell)).tolist():
                flat.append(int(v))
            offsets.append(len(flat))
        return CellArray(
            (np.asarray(flat, dtype=np.int64), np.asarray(offsets, dtype=np.int64))
        )

    def __repr__(self) -> str:
        return f"CellArray({self.GetNumberOfCells()} cells)"


class DataSet:
    """Fields shared by `vtkPolyData` and `vtkUnstructuredGrid`."""

    def __init__(self, points):
        self.points = f64(points).reshape(-1, 3) if points is not None else np.zeros((0, 3))
        self.point_data: dict[str, np.ndarray] = {}
        self.cell_data: dict[str, np.ndarray] = {}

    def GetNumberOfPoints(self) -> int:
        return int(self.points.shape[0])

    def GetNumberOfCells(self) -> int:
        raise NotImplementedError

    def GetMaxCellSize(self) -> int:
        raise NotImplementedError

    def GetPoint(self, i: int) -> np.ndarray:
        return self.points[i]

    def GetPointCells(self, i: int) -> list[np.ndarray]:
        raise NotImplementedError


class UnstructuredGrid(DataSet):
    """`vtkUnstructuredGrid`: one cell array plus a per-cell type."""

    def __init__(self, points=None, types=None, cells=()):
        super().__init__(points)
        self.types = i64(
            np.asarray(types, dtype=np.int64).ravel()
            if types is not None
            else np.zeros(0)
        )
        self.cells = CellArray(cells)

    def GetNumberOfCells(self) -> int:
        return self.cells.GetNumberOfCells()

    def GetMaxCellSize(self) -> int:
        return self.cells.max_cell_size()

    def GetCellPoints(self, cell: int) -> np.ndarray:
        return self.cells.GetCell(cell)

    def GetCellType(self, cell: int) -> int:
        return int(self.types[cell])

    def GetPointCells(self, i: int) -> list[np.ndarray]:
        out = []
        for c in range(self.GetNumberOfCells()):
            if i in self.cells.GetCell(c):
                out.append(self.cells.GetCell(c))
        return out

    def to_polydata(self) -> "PolyData":
        """`vtkPolyData` view: verts, lines, polys and strips split by type."""
        pd = PolyData(self.points)
        pd.point_data = dict(self.point_data)
        for c in range(self.GetNumberOfCells()):
            t = self.GetCellType(c)
            ids = self.cells.GetCell(c)
            if t in (VTK_VERTEX, VTK_POLY_VERTEX):
                pd.verts.append(ids)
            elif t in (VTK_LINE, VTK_POLY_LINE):
                pd.lines.append(ids)
            elif t == VTK_TRIANGLE_STRIP:
                pd.strips.append(ids)
            else:
                pd.polys.append(ids)
        pd.rebuild()
        return pd

    @staticmethod
    def from_polydata(pd: "PolyData", cell_types=None) -> "UnstructuredGrid":
        """`vtkPolyData::AppendPolygonalData` in the other direction."""
        blocks: list[np.ndarray] = []
        types: list[int] = []
        for c, ids in enumerate(pd.verts.GetCells()):
            blocks.append(ids)
            types.append(poly_cell_type(len(ids)) if cell_types is None else cell_types[0])
        for ids in pd.lines.GetCells():
            blocks.append(ids)
            types.append(VTK_POLY_LINE)
        for ids in pd.polys.GetCells():
            blocks.append(ids)
            types.append(poly_cell_type(len(ids)))
        for ids in pd.strips.GetCells():
            blocks.append(ids)
            types.append(VTK_TRIANGLE_STRIP)
        ug = UnstructuredGrid(pd.points, np.asarray(types, dtype=np.int64), ())
        conn: list[int] = []
        offsets = [0]
        for b in blocks:
            conn.extend(int(v) for v in b)
            offsets.append(len(conn))
        ug.cells = CellArray(
            (np.asarray(conn, dtype=np.int64), np.asarray(offsets, dtype=np.int64))
        )
        ug.point_data = dict(pd.point_data)
        return ug

    def __repr__(self) -> str:
        return (
            f"UnstructuredGrid({self.GetNumberOfPoints()} points, "
            f"{self.GetNumberOfCells()} cells)"
        )


class PolyData(DataSet):
    """`vtkPolyData`: four cell arrays over one point array."""

    def __init__(self, points=None):
        super().__init__(points)
        self._verts: list[np.ndarray] = []
        self._lines: list[np.ndarray] = []
        self._polys: list[np.ndarray] = []
        self._strips: list[np.ndarray] = []
        self.verts = CellArray(())
        self.lines = CellArray(())
        self.polys = CellArray(())
        self.strips = CellArray(())
        self.rebuild()

    # builders ------------------------------------------------------------
    def add_verts(self, cells) -> "PolyData":
        self._verts.extend(_as_list(cells))
        self.rebuild()
        return self

    def add_lines(self, cells) -> "PolyData":
        self._lines.extend(_as_list(cells))
        self.rebuild()
        return self

    def add_polys(self, cells) -> "PolyData":
        self._polys.extend(_as_list(cells))
        self.rebuild()
        return self

    def add_strips(self, cells) -> "PolyData":
        self._strips.extend(_as_list(cells))
        self.rebuild()
        return self

    def set_polys(self, connectivity, offsets) -> "PolyData":
        self.polys = CellArray((connectivity, offsets))
        return self

    def rebuild(self) -> "PolyData":
        self.verts = CellArray(_flatten(self._verts))
        self.lines = CellArray(_flatten(self._lines))
        self.polys = CellArray(_flatten(self._polys))
        self.strips = CellArray(_flatten(self._strips))
        return self

    # vtkPolyData accessors ----------------------------------------------
    def GetNumberOfVerts(self) -> int:
        return self.verts.GetNumberOfCells()

    def GetNumberOfLines(self) -> int:
        return self.lines.GetNumberOfCells()

    def GetNumberOfPolys(self) -> int:
        return self.polys.GetNumberOfCells()

    def GetNumberOfStrips(self) -> int:
        return self.strips.GetNumberOfCells()

    def GetNumberOfCells(self) -> int:
        return (
            self.GetNumberOfVerts()
            + self.GetNumberOfLines()
            + self.GetNumberOfPolys()
            + self.GetNumberOfStrips()
        )

    def GetMaxCellSize(self) -> int:
        return max(
            self.verts.max_cell_size(),
            self.lines.max_cell_size(),
            self.polys.max_cell_size(),
            self.strips.max_cell_size(),
        )

    def GetCellPoints(self, cell: int) -> np.ndarray:
        for ca in (self.verts, self.lines, self.polys, self.strips):
            if cell < ca.GetNumberOfCells():
                return ca.GetCell(cell)
            cell -= ca.GetNumberOfCells()
        raise IndexError(cell)

    def GetCellType(self, cell: int) -> int:
        for ca, kind in (
            (self.verts, VTK_VERTEX),
            (self.lines, VTK_LINE),
            (self.polys, None),
            (self.strips, VTK_TRIANGLE_STRIP),
        ):
            if cell < ca.GetNumberOfCells():
                if kind is not None:
                    return kind
                return poly_cell_type(ca.GetNumberOfIds(cell))
            cell -= ca.GetNumberOfCells()
        raise IndexError(cell)

    def GetCellPointsAll(self) -> list[np.ndarray]:
        return (
            self.verts.GetCells()
            + self.lines.GetCells()
            + self.polys.GetCells()
            + self.strips.GetCells()
        )

    def to_polydata(self) -> "PolyData":
        """A `vtkPolyData` is already one."""
        return self

    def to_unstructured_grid(self) -> UnstructuredGrid:
        """`vtkPolyData` as a `vtkUnstructuredGrid`: the cells in VTK order."""
        return UnstructuredGrid.from_polydata(self)

    def cell_types(self) -> np.ndarray:
        out: list[int] = []
        out.extend([VTK_VERTEX] * self.GetNumberOfVerts())
        out.extend([VTK_LINE] * self.GetNumberOfLines())
        out.extend(int(t) for t in self.polys.cell_types())
        out.extend([VTK_TRIANGLE_STRIP] * self.GetNumberOfStrips())
        return np.asarray(out, dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f"PolyData({self.GetNumberOfPoints()} points, "
            f"{self.GetNumberOfVerts()} verts, {self.GetNumberOfLines()} lines, "
            f"{self.GetNumberOfPolys()} polys, {self.GetNumberOfStrips()} strips)"
        )


def _as_list(cells) -> list[np.ndarray]:
    if cells is None:
        return []
    if isinstance(cells, CellArray):
        return cells.GetCells()
    arr = np.asarray(cells, dtype=np.int64)
    if arr.ndim == 1:
        return [arr]
    return [np.asarray(r, dtype=np.int64).ravel() for r in arr]


def _flatten(blocks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    flat: list[int] = []
    offsets = [0]
    for b in blocks:
        flat.extend(int(v) for v in np.asarray(b).ravel().tolist())
        offsets.append(len(flat))
    return (
        np.asarray(flat, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
    )


def points_of(pd: PolyData) -> np.ndarray:
    return f64(pd.points)
