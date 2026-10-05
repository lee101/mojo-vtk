"""Shared fixtures and VTK <-> mojo-vtk conversion helpers."""

from __future__ import annotations

import numpy as np
import pytest

import mojovtk as mv

vtk = pytest.importorskip("vtk", reason="upstream vtk is not installed")


def to_vtk_points(pts, double: bool = False) -> "vtk.vtkPoints":
    v = vtk.vtkPoints()
    if double:
        # vtkPoints defaults to VTK_FLOAT, which is the single most visible
        # divergence between the two libraries in the triangulator; the
        # double-precision variant isolates it.
        v.SetDataTypeToDouble()
    for p in np.asarray(pts, dtype=np.float64).reshape(-1, 3):
        v.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    return v


def to_vtk_cell_array(cells) -> "vtk.vtkCellArray":
    ca = vtk.vtkCellArray()
    for cell in cells:
        ids = [int(v) for v in np.asarray(cell).ravel()]
        if not ids:
            continue
        ca.InsertNextCell(len(ids))
        for i in ids:
            ca.InsertCellPoint(i)
    return ca


def to_vtk_polydata(pd: mv.PolyData, double: bool = False) -> "vtk.vtkPolyData":
    out = vtk.vtkPolyData()
    out.SetPoints(to_vtk_points(pd.points, double))
    for name in ("verts", "lines", "polys", "strips"):
        ca = getattr(pd, name)
        if ca.GetNumberOfCells() == 0:
            continue
        setter = {
            "verts": out.SetVerts,
            "lines": out.SetLines,
            "polys": out.SetPolys,
            "strips": out.SetStrips,
        }[name]
        setter(to_vtk_cell_array(ca.GetCells()))
    for name, arr in pd.point_data.items():
        _add_vtk_array(out.GetPointData(), name, arr, pd.GetNumberOfPoints())
    n = pd.GetNumberOfCells()
    for name, arr in pd.cell_data.items():
        _add_vtk_array(out.GetCellData(), name, arr, n)
    return out


def _add_vtk_array(attrs, name, arr, n):
    a = np.asarray(arr)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if a.dtype.kind in "iub":
        v = vtk.vtkIntArray()
        v.SetName(name)
        v.SetNumberOfComponents(int(a.shape[1]))
        for row in a:
            v.InsertNextTuple([int(x) for x in row])
    else:
        # double, so the VTK side is not limited by vtkFloatArray's float32
        v = vtk.vtkDoubleArray()
        v.SetName(name)
        v.SetNumberOfComponents(int(a.shape[1]))
        for row in a:
            v.InsertNextTuple([float(x) for x in row])
    attrs.AddArray(v)


def to_vtk_unstructured_grid(ug: mv.UnstructuredGrid) -> "vtk.vtkUnstructuredGrid":
    out = vtk.vtkUnstructuredGrid()
    out.SetPoints(to_vtk_points(ug.points))
    types = vtk.vtkUnsignedCharArray()
    for t in np.asarray(ug.types).ravel():
        types.InsertNextValue(int(t))
    out.SetCells(types, to_vtk_cell_array(ug.cells.GetCells()))
    for name, arr in ug.point_data.items():
        _add_vtk_array(out.GetPointData(), name, arr, ug.GetNumberOfPoints())
    for name, arr in ug.cell_data.items():
        _add_vtk_array(out.GetCellData(), name, arr, ug.GetNumberOfCells())
    return out


def vtk_polys(pd) -> list[np.ndarray]:
    ca = pd.GetPolys()
    out = []
    ca.InitTraversal()
    ids = vtk.vtkIdList()
    while ca.GetNextCell(ids):
        out.append(np.asarray([ids.GetId(i) for i in range(ids.GetNumberOfIds())]))
    return out


def vtk_lines(pd) -> list[np.ndarray]:
    ca = pd.GetLines()
    out = []
    ca.InitTraversal()
    ids = vtk.vtkIdList()
    while ca.GetNextCell(ids):
        out.append(np.asarray([ids.GetId(i) for i in range(ids.GetNumberOfIds())]))
    return out


def vtk_array(attrs, name) -> np.ndarray:
    a = attrs.GetArray(name)
    if a is None:
        return None
    return np.asarray(
        [list(a.GetTuple(i)) for i in range(a.GetNumberOfTuples())], dtype=np.float64
    )


def cell_array_pairs(ca: mv.CellArray) -> list[tuple]:
    return sorted(tuple(int(v) for v in ca.GetCell(c)) for c in range(ca.GetNumberOfCells()))


@pytest.fixture
def sphere():
    """A unit sphere as triangles, via VTK's own source filter."""
    src = vtk.vtkSphereSource()
    src.SetThetaResolution(24)
    src.SetPhiResolution(12)
    src.Update()
    return src.GetOutput()
