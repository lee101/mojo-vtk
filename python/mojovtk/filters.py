"""Filter classes mirroring `vtkFiltersCore`, `vtkFiltersModeling` and
`Common/ComputationalGeometry`.

Each class keeps upstream's name, its `Set*`/`Get*` spelling and its defaults,
so a VTK call site translates one-for-one.  What differs is documented in the
README: VTK builds a pipeline and calls `Update()`, here the data arrays are
plain attributes and `GetOutput()` runs the ported kernel synchronously.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, i64, lib
from .datamodel import (
    CellArray,
    PolyData,
    UnstructuredGrid,
    VTK_TRIANGLE,
    empty_cell_array,
)

# Scratch pool for the general 3-vector window `ported.mojo` carves up first;
# the per-kernel ceilings below follow the layouts documented in that file.
_TRI_W_FLOATS = 512


def _scratch_floats(n: int) -> np.ndarray:
    return np.zeros(max(int(n), 1), dtype=np.float64)


def _scratch_ints(n: int) -> np.ndarray:
    return np.zeros(max(int(n), 1), dtype=np.int64)


def _as_unstructured(data) -> UnstructuredGrid:
    """`vtkCellCenters` takes a vtkDataSet; a PolyData is flattened first."""
    if isinstance(data, PolyData):
        return data.to_unstructured_grid()
    return data


class _Filter:
    def __init__(self):
        self._input = None

    def SetInputData(self, data):
        self._input = data
        return self

    def GetInput(self):
        return self._input

    def Update(self):
        return self.GetOutput()


# ------------------------------------------------------------ vtkTriangleFilter
class vtkTriangleFilter(_Filter):
    """`Filters/Core/vtkTriangleFilter.cxx`."""

    def __init__(self):
        super().__init__()
        self.PassVerts = 1
        self.PassLines = 0
        self.PreservePolys = 0
        self.Tolerance = 0.0

    def SetPassVerts(self, v):
        self.PassVerts = int(v)

    def SetPassLines(self, v):
        self.PassLines = int(v)

    def SetPreservePolys(self, v):
        self.PreservePolys = int(v)

    def SetTolerance(self, v):
        self.Tolerance = float(v)

    def GetOutput(self) -> PolyData:
        pd = self._input
        out = PolyData(pd.points)
        out.point_data = dict(pd.point_data)

        max_cell = max(pd.GetMaxCellSize(), 3)
        n_verts = pd.verts.GetNumberOfCells()
        n_lines = pd.lines.GetNumberOfCells()
        n_polys = pd.polys.GetNumberOfCells()
        n_strips = pd.strips.GetNumberOfCells()

        # worst case: every poly fans to npts-2 triangles, every strip to npts-2
        max_o_verts = max(int(pd.verts.offsets[-1]), 1) if n_verts else 1
        max_o_lines = 2 * max(int(pd.lines.offsets[-1]), 1) if n_lines else 1
        max_o_polys = int(sum(max(int(n) - 2, 1) for n in _cell_sizes(pd.polys)))
        max_o_polys += sum(max(int(n) - 2, 1) for n in _cell_sizes(pd.strips))
        max_o_polys = max(max_o_polys, 1)

        o_vc = _scratch_ints(max_o_verts)
        o_vo = _scratch_ints(n_verts + 1)
        o_lc = _scratch_ints(max_o_lines)
        o_lo = _scratch_ints(n_lines + 1)
        o_pc = _scratch_ints(3 * max_o_polys)
        o_po = _scratch_ints(max_o_polys + 1)
        src = _scratch_ints(max_o_verts + max_o_lines + max_o_polys)
        counts = _scratch_ints(3)
        sf = _scratch_floats(_TRI_W_FLOATS + 12 * max_cell)
        si = _scratch_ints(16 + 8 * max_cell)

        # scratchI layout inside TriangleFilter:
        # state[2] queueState[3] queueId[M] itemLoc[M] tris[3M] trisCount[1]
        # connOffset[1]
        si[:] = 0
        queueState = si[2:5]
        queueState[2] = max_cell

        tolerance = np.asarray([self.Tolerance], dtype=np.float64)
        lib().mvt_triangle_filter(
            addr(pd.points), addr(pd.verts.connectivity), addr(pd.verts.offsets), n_verts,
            addr(pd.lines.connectivity), addr(pd.lines.offsets), n_lines,
            addr(pd.polys.connectivity), addr(pd.polys.offsets), n_polys,
            addr(pd.strips.connectivity), addr(pd.strips.offsets), n_strips,
            int(self.PassVerts), int(self.PassLines), int(self.PreservePolys),
            addr(tolerance), max_cell,
            addr(o_vc), addr(o_vo), addr(o_lc), addr(o_lo), addr(o_pc), addr(o_po),
            addr(src), addr(counts), addr(sf), addr(si),
        )
        nv, nl, npoly = (int(counts[0]), int(counts[1]), int(counts[2]))

        v0, l0, p0 = 0, nv, nv + nl
        out.verts = CellArray((o_vc[: o_vo[nv]], o_vo[: nv + 1]))
        out.lines = CellArray((o_lc[: o_lo[nl]], o_lo[: nl + 1]))
        out.polys = CellArray((o_pc[: o_po[npoly]], o_po[: npoly + 1]))
        srcs = {
            name: np.asarray(arr)[src[l0 : l0 + nl]]
            for name, arr in pd.cell_data.items()
        }
        srcs.update(
            {name: np.asarray(arr)[src[p0 : p0 + npoly]] for name, arr in pd.cell_data.items()}
        )
        out.cell_data = srcs
        return out


def _cell_sizes(ca: CellArray) -> np.ndarray:
    if ca.GetNumberOfCells() == 0:
        return np.zeros(0, dtype=np.int64)
    return np.diff(ca.offsets.astype(np.int64))


# ------------------------------------------------------------- vtkCellCenters
class vtkCellCenters(_Filter):
    """`Filters/Core/vtkCellCenters.cxx`."""

    def __init__(self):
        super().__init__()
        self.VertexCells = 0
        self.CopyArrays = 1

    def SetVertexCells(self, v):
        self.VertexCells = int(v)

    def SetCopyArrays(self, v):
        self.CopyArrays = int(v)

    def GetOutput(self) -> PolyData:
        ug = _as_unstructured(self._input)
        n_cells = ug.GetNumberOfCells()
        centers = np.zeros((max(n_cells, 1), 3), dtype=np.float64)
        cell_centered = np.zeros(max(n_cells, 1), dtype=np.int64)
        num_points = _scratch_ints(1)
        max_cell = max(ug.GetMaxCellSize(), 8)
        sf = _scratch_floats(256 + 3 * max_cell)
        si = _scratch_ints(8)

        lib().mvt_cell_centers(
            addr(ug.points), addr(ug.cells.connectivity), addr(ug.cells.offsets),
            addr(ug.types), n_cells, addr(centers), addr(cell_centered),
            addr(num_points), addr(sf), addr(si),
        )
        n = int(num_points[0])
        pts = centers[:n]
        out = PolyData(pts)
        if self.CopyArrays:
            out.point_data = {
                name: np.asarray(arr)[:n] for name, arr in ug.cell_data.items()
            }
        if self.VertexCells:
            ids = np.arange(n, dtype=np.int64)
            out.verts = CellArray((ids, np.arange(n + 1, dtype=np.int64)))
            out.cell_data = dict(out.point_data)
        return out


# -------------------------------------------------------- vtkMassProperties
class vtkMassProperties(_Filter):
    """`Filters/Core/vtkMassProperties.cxx`."""

    def __init__(self):
        super().__init__()
        self._result = None

    def _run(self):
        pd = self._input
        result = _scratch_floats(256)
        lib().mvt_mass_properties(
            addr(pd.points), addr(pd.polys.connectivity), addr(pd.polys.offsets),
            pd.GetNumberOfPolys(), pd.GetNumberOfPoints(), addr(result),
        )
        self._result = result

    def GetVolume(self) -> float:
        self._run()
        return float(self._result[9])

    def GetVolumeProjected(self) -> float:
        self._run()
        return float(self._result[10])

    def GetVolumeX(self) -> float:
        self._run()
        return float(self._result[3])

    def GetVolumeY(self) -> float:
        self._run()
        return float(self._result[4])

    def GetVolumeZ(self) -> float:
        self._run()
        return float(self._result[5])

    def GetKx(self) -> float:
        self._run()
        return float(self._result[6])

    def GetKy(self) -> float:
        self._run()
        return float(self._result[7])

    def GetKz(self) -> float:
        self._run()
        return float(self._result[8])

    def GetSurfaceArea(self) -> float:
        self._run()
        return float(self._result[0])

    def GetMinCellArea(self) -> float:
        self._run()
        return float(self._result[1])

    def GetMaxCellArea(self) -> float:
        self._run()
        return float(self._result[2])

    def GetNormalizedShapeIndex(self) -> float:
        self._run()
        return float(self._result[11])


# ------------------------------------------------------------ vtkFeatureEdges
class vtkFeatureEdges(_Filter):
    """`Filters/Core/vtkFeatureEdges.cxx`."""

    def __init__(self):
        super().__init__()
        self.FeatureAngle = 30.0
        self.BoundaryEdges = 1
        self.FeatureEdges = 1
        self.NonManifoldEdges = 1
        self.ManifoldEdges = 0
        self.PassLines = 0
        self.Coloring = 1
        self.RemoveGhostInterfaces = 1

    def SetFeatureAngle(self, v):
        self.FeatureAngle = float(v)

    def SetBoundaryEdges(self, v):
        self.BoundaryEdges = int(v)

    def SetFeatureEdges(self, v):
        self.FeatureEdges = int(v)

    def SetNonManifoldEdges(self, v):
        self.NonManifoldEdges = int(v)

    def SetManifoldEdges(self, v):
        self.ManifoldEdges = int(v)

    def SetPassLines(self, v):
        self.PassLines = int(v)

    def SetColoring(self, v):
        self.Coloring = int(v)

    def GetOutput(self) -> PolyData:
        pd = self._input
        n_polys = pd.GetNumberOfPolys()
        n_edges = int(_cell_sizes(pd.polys).sum()) if n_polys else 0
        # must match the power-of-two capacity FeatureEdges builds for its table
        cap = 4
        while cap < 2 * n_edges + 2:
            cap *= 2
        max_lines = max(n_edges, 1)
        new_pts = np.zeros((2 * max_lines, 3), dtype=np.float64)
        conn = _scratch_ints(2 * max_lines)
        offs = _scratch_ints(max_lines + 1)
        edge_types = np.zeros(max_lines, dtype=np.float64)
        poly_normals = np.zeros((max(n_polys, 1), 3), dtype=np.float64)
        src = _scratch_ints(max_lines)
        counts = _scratch_ints(6)
        sf = _scratch_floats(32)
        si = _scratch_ints(
            2 * cap + 2 * n_edges + n_edges + pd.GetNumberOfPoints() + 8
        )

        angle = np.asarray([self.FeatureAngle], dtype=np.float64)
        lib().mvt_feature_edges(
            addr(pd.points), addr(pd.polys.connectivity), addr(pd.polys.offsets),
            n_polys, pd.GetNumberOfPoints(), addr(angle),
            int(self.BoundaryEdges), int(self.FeatureEdges), int(self.NonManifoldEdges),
            int(self.ManifoldEdges), int(self.Coloring),
            addr(new_pts), addr(conn), addr(offs), addr(edge_types), addr(poly_normals),
            addr(src), addr(counts), addr(sf), addr(si),
        )
        n_pts, n_lines = int(counts[0]), int(counts[1])
        self._counts = counts
        out = PolyData(new_pts[:n_pts])
        out.lines = CellArray((conn[: 2 * n_lines], offs[: n_lines + 1]))
        if self.Coloring:
            # upstream stores this in a vtkFloatArray
            out.cell_data["Edge Types"] = edge_types[:n_lines].astype(np.float32)
        out.point_data = {
            name: np.asarray(arr)[src[:n_lines]]
            for name, arr in pd.point_data.items()
        }
        return out

    def GetNumberOfBoundaryEdges(self) -> int:
        self.GetOutput()
        return self._counts[2]

    def GetNumberOfNonManifoldEdges(self) -> int:
        self.GetOutput()
        return self._counts[3]

    def GetNumberOfFeatureEdges(self) -> int:
        self.GetOutput()
        return self._counts[4]

    def GetNumberOfManifoldEdges(self) -> int:
        self.GetOutput()
        return self._counts[5]


# --------------------------------------------------------- vtkPolyDataNormals
class vtkPolyDataNormals(_Filter):
    """`Filters/Core/vtkPolyDataNormals.cxx` (`GetCellNormals`/`GetPointNormals`).

    Splitting and consistency are not ported: they are separate upstream
    filters (vtkSplitSharpEdgesPolyData, vtkOrientPolyData) and this port runs
    the non-splitting path.  See the README.
    """

    def __init__(self):
        super().__init__()
        self.FeatureAngle = 30.0
        self.Splitting = 1
        self.Consistency = 1
        self.FlipNormals = 0
        self.ComputePointNormals = 1
        self.ComputeCellNormals = 0
        self.NonManifoldTraversal = 1
        self.AutoOrientNormals = 0

    def SetFeatureAngle(self, v):
        self.FeatureAngle = float(v)

    def SetSplitting(self, v):
        self.Splitting = int(v)

    def SetConsistency(self, v):
        self.Consistency = int(v)

    def SetFlipNormals(self, v):
        self.FlipNormals = int(v)

    def SetComputePointNormals(self, v):
        self.ComputePointNormals = int(v)

    def SetComputeCellNormals(self, v):
        self.ComputeCellNormals = int(v)

    def GetOutput(self) -> PolyData:
        pd = self._input
        n_verts = pd.GetNumberOfVerts()
        n_lines = pd.GetNumberOfLines()
        n_polys = pd.GetNumberOfPolys()
        n_strips = pd.GetNumberOfStrips()
        n_cells = n_verts + n_lines + n_polys + n_strips
        cell_normals = np.zeros((max(n_cells, 1), 3), dtype=np.float64)
        sf = _scratch_floats(16)
        lib().mvt_polydata_normals_cell_normals(
            addr(pd.points), n_verts, n_lines, addr(pd.polys.connectivity),
            addr(pd.polys.offsets), n_polys, n_strips, addr(cell_normals), addr(sf),
        )
        # upstream stores normals in a vtkFloatArray
        out = PolyData(pd.points)
        out.verts = CellArray((pd.verts.connectivity, pd.verts.offsets))
        out.lines = CellArray((pd.lines.connectivity, pd.lines.offsets))
        out.polys = CellArray((pd.polys.connectivity, pd.polys.offsets))
        out.strips = CellArray((pd.strips.connectivity, pd.strips.offsets))
        out.point_data = dict(pd.point_data)
        out.cell_data = dict(pd.cell_data)
        if self.ComputeCellNormals:
            out.cell_data["Normals"] = cell_normals[:n_cells].astype(np.float32)
        if self.ComputePointNormals:
            n_pts = pd.GetNumberOfPoints()
            point_normals = np.zeros((max(n_pts, 1), 3), dtype=np.float64)
            flip_direction = -1.0 if (self.FlipNormals and not self.Consistency) else 1.0
            link_offsets = _scratch_ints(n_pts + 1)
            link_cells = _scratch_ints(max(int(_cell_sizes(pd.polys).sum()), 1))
            cursor = _scratch_ints(max(n_pts, 1))
            flip = np.asarray([flip_direction], dtype=np.float64)
            lib().mvt_polydata_normals_point_normals(
                addr(pd.points), addr(pd.polys.connectivity), addr(pd.polys.offsets),
                n_polys, n_pts, addr(cell_normals), addr(point_normals), addr(flip),
                addr(link_offsets), addr(link_cells), addr(cursor),
            )
            out.point_data["Normals"] = point_normals[:n_pts].astype(np.float32)
        return out


# ---------------------------------------------------- vtkPointDataToCellData
class vtkPointDataToCellData(_Filter):
    """`Filters/Core/vtkPointDataToCellData.cxx`."""

    def __init__(self):
        super().__init__()
        self.CategoricalData = 0
        self.CategoricalComponent = 0

    def SetCategoricalData(self, v):
        self.CategoricalData = int(v)

    def SetCategoricalComponent(self, v):
        self.CategoricalComponent = int(v)

    def GetOutput(self) -> UnstructuredGrid:
        ug = self._input
        n_cells = ug.GetNumberOfCells()
        out = UnstructuredGrid(ug.points, ug.types, (ug.cells.connectivity, ug.cells.offsets))
        out.point_data = dict(ug.point_data)
        out.cell_data = dict(ug.cell_data)
        for name, arr in ug.point_data.items():
            a = f64(arr)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            ncomp = a.shape[1]
            dst = np.zeros((max(n_cells, 1), ncomp), dtype=np.float64)
            has_data = np.zeros(max(n_cells, 1), dtype=np.int64)
            if self.CategoricalData:
                bins = np.zeros(3 * (ug.GetMaxCellSize() + 1), dtype=np.float64)
                lib().mvt_point_data_to_cell_data_categorical(
                    addr(a), addr(ug.cells.connectivity), addr(ug.cells.offsets),
                    n_cells, addr(dst), addr(has_data), addr(bins),
                )
            else:
                lib().mvt_point_data_to_cell_data(
                    addr(a), ncomp, addr(ug.cells.connectivity), addr(ug.cells.offsets),
                    n_cells, addr(dst), addr(has_data),
                )
            out.cell_data[name] = dst[:n_cells]
        return out


# ---------------------------------------------------- vtkCellDataToPointData
class vtkCellDataToPointData(_Filter):
    """`Filters/Core/vtkCellDataToPointData.cxx` (the `All` fast path)."""

    def __init__(self):
        super().__init__()
        self.PassCellData = 0

    def SetPassCellData(self, v):
        self.PassCellData = int(v)

    def GetOutput(self) -> UnstructuredGrid:
        ug = self._input
        n_pts = ug.GetNumberOfPoints()
        n_cells = ug.GetNumberOfCells()
        out = UnstructuredGrid(ug.points, ug.types, (ug.cells.connectivity, ug.cells.offsets))
        out.cell_data = dict(ug.cell_data)
        link_offsets = _scratch_ints(n_pts + 1)
        link_cells = _scratch_ints(max(int(_cell_sizes(ug.cells).sum()), 1))
        cursor = _scratch_ints(max(n_pts, 1))
        for name, arr in ug.cell_data.items():
            a = f64(arr)
            if a.ndim == 1:
                a = a.reshape(-1, 1)
            ncomp = a.shape[1]
            dst = np.zeros((max(n_pts, 1), ncomp), dtype=np.float64)
            lib().mvt_cell_data_to_point_data(
                addr(a), ncomp, addr(ug.cells.connectivity), addr(ug.cells.offsets),
                n_cells, n_pts, addr(dst), addr(link_offsets), addr(link_cells),
                addr(cursor),
            )
            out.point_data[name] = dst[:n_pts]
        if self.PassCellData:
            out.cell_data = dict(ug.cell_data)
        return out


# ------------------------------------------------------- vtkCardinalSpline
class vtkCardinalSpline:
    """`Common/ComputationalGeometry/vtkCardinalSpline.cxx`."""

    def __init__(self):
        self._ts: np.ndarray | None = None
        self._cache = None
        self._stamp = None
        self.Closed = 0
        self.LeftConstraint = 1
        self.LeftValue = 0.0
        self.RightConstraint = 1
        self.RightValue = 0.0
        # vtkSpline::ParametricRange starts at {-1, -1}, which means "unset"
        self.ParametricRange = [-1.0, -1.0]

    def SetCurve(self, ts, ys):
        """Bulk load; `AddPoint` is the upstream spelling and does the same."""
        ts = f64(ts).ravel()
        ys = f64(ys).ravel()
        pair = np.empty(2 * ts.size, dtype=np.float64)
        pair[0::2] = ts
        pair[1::2] = ys
        self._ts = pair
        self._stamp = None
        return self

    def AddPoint(self, x: float, y: float):
        """`vtkSpline::AddPoint`, including its parametric-range clamp."""
        if self.ParametricRange[0] != self.ParametricRange[1]:
            x = min(max(x, self.ParametricRange[0]), self.ParametricRange[1])
        if self._ts is None or self._ts.size == 0:
            self.SetCurve(np.asarray([x]), np.asarray([y]))
            return self
        pair = np.empty(self._ts.size + 2, dtype=np.float64)
        pair[:-2] = self._ts
        pair[-2] = x
        pair[-1] = y
        self._ts = pair
        self._stamp = None
        return self

    def SetPoint(self, i: int, x: float, y: float):
        """`vtkSpline::SetPoint`."""
        if self._ts is None:
            raise ValueError("no points")
        self._ts[2 * i] = x
        self._ts[2 * i + 1] = y
        self._stamp = None
        return self

    def GetNumberOfPoints(self) -> int:
        return 0 if self._ts is None else int(self._ts.size // 2)

    def GetPoints(self) -> np.ndarray:
        return np.zeros((0, 2), dtype=np.float64)

    def SetClosed(self, v):
        self.Closed = int(v)
        self._stamp = None

    def SetParametricRange(self, lo, hi):
        self.ParametricRange = [float(lo), float(hi)]
        self._stamp = None

    def SetLeftConstraint(self, v):
        self.LeftConstraint = int(v)
        self._stamp = None

    def SetLeftValue(self, v):
        self.LeftValue = float(v)
        self._stamp = None

    def SetRightConstraint(self, v):
        self.RightConstraint = int(v)
        self._stamp = None

    def SetRightValue(self, v):
        self.RightValue = float(v)
        self._stamp = None

    def _compute(self):
        """`vtkCardinalSpline::Compute`, run only when the curve has changed."""
        if self._ts is None:
            raise ValueError("SetCurve has not been called")
        size = int(self._ts.size // 2)
        stamp = (
            size,
            self.Closed,
            tuple(self.ParametricRange),
            self.LeftConstraint,
            self.LeftValue,
            self.RightConstraint,
            self.RightValue,
        )
        if getattr(self, "_stamp", None) == stamp:
            return self._cache
        self._stamp = stamp
        closed = bool(self.Closed)
        n = size + 1 if closed else size
        intervals = np.zeros(max(n, 1), dtype=np.float64)
        work = np.zeros(max(n, 1), dtype=np.float64)
        dependent = np.zeros(max(n, 1), dtype=np.float64)
        coefficients = np.zeros(4 * max(n, 1), dtype=np.float64)
        prange = np.asarray(self.ParametricRange, dtype=np.float64)
        lv = np.asarray([self.LeftValue], dtype=np.float64)
        rv = np.asarray([self.RightValue], dtype=np.float64)
        got = lib().mvt_cardinal_spline_compute(
            addr(self._ts), addr(self._ts), size, int(closed), addr(prange),
            addr(intervals), addr(work), addr(dependent), addr(coefficients),
            self.LeftConstraint, addr(lv), self.RightConstraint, addr(rv),
        )
        self._cache = (intervals, coefficients, int(got))
        return self._cache

    def Evaluate(self, t: float) -> float:
        intervals, coefficients, size = self._compute()
        tt = np.asarray([t], dtype=np.float64)
        return float(
            lib().mvt_cardinal_spline_evaluate(
                addr(intervals), addr(coefficients), size, addr(tt)
            )
        )


# ------------------------------------------------ vtkLoopSubdivisionFilter
class vtkLoopSubdivisionFilter(_Filter):
    """`Filters/Modeling/vtkLoopSubdivisionFilter.cxx`."""

    def __init__(self):
        super().__init__()
        self.NumberOfSubdivisions = 1
        self.CheckForTriangles = 1

    def SetNumberOfSubdivisions(self, v):
        self.NumberOfSubdivisions = int(v)

    def GetOutput(self) -> PolyData:
        pd = self._input
        # vtkSubdivisionFilter::RequestData: only triangles may be subdivided
        if self.CheckForTriangles:
            for t in pd.cell_types():
                if t != VTK_TRIANGLE:
                    raise ValueError(
                        "vtkLoopSubdivisionFilter only operates on triangles"
                    )
        n_pts = pd.GetNumberOfPoints()
        n_cells = pd.GetNumberOfCells()
        n_edges = int(_cell_sizes(pd.polys).sum()) if n_cells else 0
        cap = 1
        while cap < 2 * n_edges + 2:
            cap *= 2
        max_pts = max(n_pts + n_edges, 1)
        max_cells = max(4 * n_cells, 1)

        out_pts = np.zeros((max_pts, 3), dtype=np.float64)
        conn = _scratch_ints(3 * max_cells)
        offs = _scratch_ints(max_cells + 1)
        src = _scratch_ints(max_cells)
        edge_data = _scratch_ints(3 * max(n_cells, 1))
        counts = _scratch_ints(2)
        sf = _scratch_floats(512)
        si = _scratch_ints(
            4 * cap + 4 * n_edges + 2 * n_pts + 3 * (n_cells + 4) + 8
        )

        level = pd
        for _ in range(self.NumberOfSubdivisions):
            nc = level.GetNumberOfCells()
            if nc == 0 or level.GetNumberOfPoints() == 0:
                return PolyData(np.zeros((0, 3)))
            ne = int(_cell_sizes(level.polys).sum())
            c = 1
            while c < 2 * ne + 2:
                c *= 2
            out_pts = np.zeros((max(level.GetNumberOfPoints() + ne, 1), 3))
            conn = _scratch_ints(3 * max(4 * nc, 1))
            offs = _scratch_ints(4 * nc + 1)
            src = _scratch_ints(4 * nc)
            edge_data = _scratch_ints(3 * nc)
            si = _scratch_ints(
                4 * c + 4 * ne + 2 * level.GetNumberOfPoints() + 3 * (nc + 4) + 8
            )
            ok = lib().mvt_loop_subdivision(
                addr(level.points), addr(level.polys.connectivity),
                addr(level.polys.offsets), nc, level.GetNumberOfPoints(),
                addr(out_pts), addr(edge_data), addr(conn), addr(offs), addr(src),
                addr(counts), addr(sf), addr(si),
            )
            if not ok:
                raise ValueError("Dataset is non-manifold and cannot be subdivided.")
            n_new_pts, n_new_cells = int(counts[0]), int(counts[1])
            nxt = PolyData(out_pts[:n_new_pts])
            nxt.polys = CellArray((conn[: 3 * n_new_cells], offs[: n_new_cells + 1]))
            nxt.cell_data = {
                name: np.asarray(arr)[src[:n_new_cells]]
                for name, arr in level.cell_data.items()
            }
            nxt.point_data = {
                name: np.asarray(arr)[:n_new_pts]
                for name, arr in level.point_data.items()
            }
            level = nxt
        return level
