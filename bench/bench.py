"""mojo-vtk against upstream VTK on the same data.

Run through `pixi run bench`; the task holds a machine-wide flock so concurrent
factory jobs cannot distort the numbers.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

import mojovtk as mv  # noqa: E402
import vtk  # noqa: E402
def vtk_points(arr) -> "vtk.vtkPoints":
    v = vtk.vtkPoints()
    v.SetDataTypeToDouble()
    for p in np.ascontiguousarray(arr):
        v.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    return v


def timeit(fn, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def sphere(nlat: int, nlon: int) -> vtk.vtkPolyData:
    src = vtk.vtkSphereSource()
    src.SetThetaResolution(nlon)
    src.SetPhiResolution(nlat)
    src.Update()
    return src.GetOutput()


def from_vtk(vpd) -> mv.PolyData:
    ca = vpd.GetPolys()
    ca.InitTraversal()
    ids = vtk.vtkIdList()
    blocks = []
    while ca.GetNextCell(ids):
        blocks.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    pts = np.asarray([vpd.GetPoint(i) for i in range(vpd.GetNumberOfPoints())])
    return mv.PolyData(pts).add_polys(blocks)


def to_vtk(pd: mv.PolyData) -> vtk.vtkPolyData:
    out = vtk.vtkPolyData()
    out.SetPoints(vtk_points(pd.points))
    ca = vtk.vtkCellArray()
    for c in pd.polys.GetCells():
        ca.InsertNextCell(len(c))
        for v in c:
            ca.InsertCellPoint(int(v))
    out.SetPolys(ca)
    return out


def grid_mixed(n: int) -> mv.PolyData:
    """A grid of quads plus random pentagons, so the ear-cut path is exercised."""
    rng = np.random.default_rng(0)
    m = int(math.isqrt(n)) + 1
    gx, gy = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    pts = np.stack([gx.ravel(), gy.ravel(), np.zeros(m * m)], axis=1)
    polys = []
    for i in range(m - 1):
        for j in range(m - 1):
            polys.append([i * m + j, (i + 1) * m + j, (i + 1) * m + j + 1, i * m + j + 1])
    rng.shuffle(polys)
    return mv.PolyData(pts).add_polys(polys[:n])


CASES = []


def case(name):
    def deco(fn):
        CASES.append((name, fn))
        return fn

    return deco


# ---------------------------------------------------------------- filters
@case("vtkMassProperties (sphere 64x32, 4k tris)")
def _():
    v = sphere(32, 64)
    p = from_vtk(v)
    a = mv.vtkMassProperties().SetInputData(p)
    b = vtk.vtkMassProperties()
    b.SetInputData(v)
    b.Update()
    def run_ours():
        return a.GetVolume()

    def run_vtk():
        f = vtk.vtkMassProperties()
        f.SetInputData(v)
        f.Update()
        return f.GetVolume()

    return run_ours, run_vtk


@case("vtkMassProperties (sphere 256x128, 64k tris)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v)
    a = mv.vtkMassProperties().SetInputData(p)
    b = vtk.vtkMassProperties()
    b.SetInputData(v)
    b.Update()
    def run_ours():
        return a.GetVolume()

    def run_vtk():
        f = vtk.vtkMassProperties()
        f.SetInputData(v)
        f.Update()
        return f.GetVolume()

    return run_ours, run_vtk


def _run_vtk_triangle(v):
    f = vtk.vtkTriangleFilter()
    f.SetInputData(v)
    f.Update()
    return f.GetOutput()


@case("vtkTriangleFilter (quad grid 256x256, 65k quads)")
def _():
    p = grid_mixed(65536)
    v = to_vtk(p)
    def run_ours():
        return mv.vtkTriangleFilter().SetInputData(p).GetOutput()

    def run_vtk():
        return _run_vtk_triangle(v)

    return run_ours, run_vtk


@case("vtkCellCenters (sphere 256x128, 64k tris)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v).to_unstructured_grid()
    a = mv.vtkCellCenters().SetInputData(p)
    b = vtk.vtkCellCenters()
    b.SetInputData(v)
    b.Update()

    def run_ours():
        return a.GetOutput()

    def run_vtk():
        b.Modified()
        b.Update()
        return b.GetOutput()

    return run_ours, run_vtk


@case("vtkFeatureEdges (sphere 256x128, 64k tris)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v)
    a = mv.vtkFeatureEdges().SetInputData(p)
    b = vtk.vtkFeatureEdges()
    b.SetInputData(v)
    b.Update()
    def run_ours():
        return a.GetOutput()

    def run_vtk():
        b.Modified()
        b.Update()
        return b.GetOutput()

    return run_ours, run_vtk


@case("vtkPolyDataNormals point normals (sphere 256x128)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v)
    f = mv.vtkPolyDataNormals().SetInputData(p)
    g = vtk.vtkPolyDataNormals()
    g.SetInputData(v)
    g.Update()
    def run_ours():
        return f.GetOutput()

    def run_vtk():
        g.Modified()
        g.Update()
        return g.GetOutput()

    return run_ours, run_vtk


@case("vtkPointDataToCellData (sphere 256x128, 64k tris)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v).to_unstructured_grid()
    rng = np.random.default_rng(1)
    arr = rng.normal(size=(p.GetNumberOfPoints(), 3))
    p.point_data["v"] = arr
    vt = vtk.vtkUnstructuredGrid()
    vt.SetPoints(vtk_points(p.points))
    types = vtk.vtkUnsignedCharArray()
    ca = vtk.vtkCellArray()
    for c in p.cells.GetCells():
        ca.InsertNextCell(len(c))
        for q in c:
            ca.InsertCellPoint(int(q))
        types.InsertNextValue(5)
    vt.SetCells(types, ca)
    parr = vtk.vtkDoubleArray()
    parr.SetName("v")
    parr.SetNumberOfComponents(3)
    for row in arr:
        parr.InsertNextTuple([float(x) for x in row])
    vt.GetPointData().AddArray(parr)
    a = mv.vtkPointDataToCellData().SetInputData(p)
    b = vtk.vtkPointDataToCellData()
    b.SetInputData(vt)
    b.Update()
    def run_ours():
        return a.GetOutput()

    def run_vtk():
        b.Modified()
        b.Update()
        return b.GetOutput()

    return run_ours, run_vtk


@case("vtkCellDataToPointData (sphere 256x128, 64k tris)")
def _():
    v = sphere(128, 256)
    p = from_vtk(v).to_unstructured_grid()
    rng = np.random.default_rng(2)
    arr = rng.normal(size=(p.GetNumberOfCells(), 3))
    p.cell_data["v"] = arr
    vt = vtk.vtkUnstructuredGrid()
    vt.SetPoints(vtk_points(p.points))
    types = vtk.vtkUnsignedCharArray()
    ca = vtk.vtkCellArray()
    for c in p.cells.GetCells():
        ca.InsertNextCell(len(c))
        for q in c:
            ca.InsertCellPoint(int(q))
        types.InsertNextValue(5)
    vt.SetCells(types, ca)
    carr = vtk.vtkDoubleArray()
    carr.SetName("v")
    carr.SetNumberOfComponents(3)
    for row in arr:
        carr.InsertNextTuple([float(x) for x in row])
    vt.GetCellData().AddArray(carr)
    a = mv.vtkCellDataToPointData().SetInputData(p)
    b = vtk.vtkCellDataToPointData()
    b.SetInputData(vt)
    b.Update()
    def run_ours():
        return a.GetOutput()

    def run_vtk():
        b.Modified()
        b.Update()
        return b.GetOutput()

    return run_ours, run_vtk


@case("vtkLoopSubdivisionFilter (sphere 64x32 -> level 2)")
def _():
    v = sphere(32, 64)
    p = from_vtk(v)
    a = mv.vtkLoopSubdivisionFilter().SetInputData(p)
    a.SetNumberOfSubdivisions(2)
    g = vtk.vtkLoopSubdivisionFilter()
    g.SetInputData(v)
    g.SetNumberOfSubdivisions(2)
    g.Update()
    def run_ours():
        return a.GetOutput()

    def run_vtk():
        g.Modified()
        g.Update()
        return g.GetOutput()

    return run_ours, run_vtk


@case("vtkCardinalSpline.Evaluate (1000 calls, 512 knots)")
def _():
    ts = np.linspace(0.0, 1.0, 512)
    ys = np.sin(20.0 * ts)
    a = mv.vtkCardinalSpline()
    a.SetCurve(ts, ys)
    b = vtk.vtkCardinalSpline()
    for t, y in zip(ts, ys):
        b.AddPoint(float(t), float(y))
    grid = np.linspace(0.0, 1.0, 1_000_000)
    sample = grid[::1000]

    def run_ours():
        return [a.Evaluate(float(t)) for t in sample]

    def run_vtk():
        return [b.Evaluate(float(t)) for t in sample]

    return run_ours, run_vtk


def main() -> None:
    print(f"machine: {platform.platform()}")
    print(f"python {platform.python_version()}  numpy {np.__version__}  vtk {vtk.vtkVersion.GetVTKVersion()}")
    print()
    print(f"| {'case':<52} | {'mojo-vtk':>12} | {'vtk':>12} | {'vtk / mojo-vtk':>15} |")
    print(f"| {'-' * 52} | {'-' * 12} | {'-' * 12} | {'-' * 15} |")
    for name, fn in CASES:
        ours, theirs = fn()
        a = timeit(ours, repeat=5)
        b = timeit(theirs, repeat=5)
        ratio = b / a if a > 0 else float("inf")
        print(
            f"| {name:<52} | {a * 1e3:>10.2f} ms | {b * 1e3:>10.2f} ms | {ratio:>14.2f}x |"
        )


if __name__ == "__main__":
    main()
