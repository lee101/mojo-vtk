"""Every `@export`ed kernel called by name, with an assertion on its real output.

`tests/test_parity.py` reaches the kernels through the public filter API, which
is how a caller meets them.  That is enough to keep a kernel honest only if the
wrapper happens to call it, so this file calls every symbol in
`src/capi.mojo` directly and checks what it wrote.  Each test still checks
parity against upstream VTK wherever upstream can express the same thing.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import mojovtk as mv
from mojovtk._lib import LIB, addr, lib

vtk = pytest.importorskip("vtk", reason="upstream vtk is not installed")

from conftest import (  # noqa: E402
    to_vtk_cell_array,
    to_vtk_points,
    vtk_polys,
)
from test_parity import _total_area, _tri_area  # noqa: E402


@pytest.fixture(autouse=True)
def _loaded_library():
    """Fail loudly, not quietly, if the compiled library is missing.

    Every assertion below is about what the Mojo kernel wrote into a buffer it
    was handed.  If the shared library never loaded, the buffers would still be
    zeros and the tests would pass for the wrong reason.
    """
    lib()
    assert lib()._name is not None


def _f(n: int) -> np.ndarray:
    return np.zeros(max(int(n), 1), dtype=np.float64)


def _i(n: int) -> np.ndarray:
    return np.zeros(max(int(n), 1), dtype=np.int64)


def _padded(n: int, dtype) -> np.ndarray:
    """A buffer with guard slack after the live data.

    Every size below is computed from the layout comments in `ported.mojo`.
    Some of those layouts index one entry past the live data (a run
    terminator, a `size + 1` offset), and NumPy hands out 16-byte-aligned
    blocks with no slack, so an overrun lands in the allocator's own metadata
    and aborts the process instead of failing a test.  This keeps the overrun
    inside the allocation.
    """
    return np.zeros(max(int(n), 1) + 32, dtype=dtype)


def _polygon_normal(pts: np.ndarray) -> np.ndarray:
    """`vtkPolygon::ComputeNormal` for an outline given as explicit points."""
    poly = vtk.vtkPolygon()
    poly.GetPoints().SetDataTypeToDouble()
    for p in pts:
        poly.GetPoints().InsertNextPoint(*p)
    poly.GetPointIds().SetNumberOfIds(len(pts))
    for i in range(len(pts)):
        poly.GetPointIds().SetId(i, i)
    poly.ComputeNormal()
    return np.asarray(poly.GetNormal(), dtype=np.float64)


def _polygon_of(points, ids) -> vtk.vtkPolygon:
    """A `vtkPolygon` over explicit points, double precision."""
    poly = vtk.vtkPolygon()
    poly.GetPoints().SetDataTypeToDouble()
    for p in points:
        poly.GetPoints().InsertNextPoint(*p)
    poly.GetPointIds().SetNumberOfIds(len(ids))
    for i, v in enumerate(ids):
        poly.GetPointIds().SetId(i, int(v))
    return poly


def _polygon_normal(points, ids=None) -> tuple:
    """`vtkPolygon::ComputeNormal` over an explicit point list.

    Returns `(vtkCellStatus, normal)`.  The `numPts/double*` overload is the
    same `NormalWorker` the port transcribes, so it is the one compared against.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if ids is not None:
        pts = pts[np.asarray(ids, dtype=np.int64)]
    out = [0.0, 0.0, 0.0]
    status = vtk.vtkPolygon.ComputeNormal(len(pts), pts.reshape(-1).tolist(), out)
    return int(status), np.asarray(out, dtype=np.float64)


# ------------------------------------------------- mvt_polygon_compute_normal
def test_mvt_polygon_compute_normal():
    """`vtkPolygon::ComputeNormal` = the anonymous-namespace `NormalWorker`."""
    # A planar pentagon traversed counter-clockwise about +z.
    pts = np.asarray(
        [[0, 0, 0], [1, 0, 0], [2, 1, 0], [1, 2, 0], [0, 1, 0]], dtype=np.float64
    )
    ids = np.arange(5, dtype=np.int64)
    normal = _f(3)
    scratch = _f(16)
    rc = lib().mvt_polygon_compute_normal(
        addr(pts), 5, addr(ids), 0, addr(normal), addr(scratch)
    )
    assert rc == 0  # vtkCellStatus::Valid
    assert np.allclose(normal, [0.0, 0.0, 1.0], atol=1e-12)

    # The same outline in reverse order must flip the normal.
    rev = pts[::-1].copy()
    lib().mvt_polygon_compute_normal(
        addr(rev), 5, addr(ids), 0, addr(normal), addr(scratch)
    )
    assert np.allclose(normal, [0.0, 0.0, -1.0], atol=1e-12)

    # And it must agree with vtkPolygon on the same points.
    status, theirs = _polygon_normal(pts[::-1].copy())
    assert status == 0  # Valid
    assert np.allclose(normal, theirs, atol=1e-12)
    assert np.allclose(_polygon_normal(pts)[1], [0.0, 0.0, 1.0], atol=1e-12)

    # `useIds` set: the ids pick the outline out of a larger point set.
    big = np.vstack([pts, [[5.0, 5.0, 5.0]]])
    sub = np.asarray([4, 3, 2, 1, 0], dtype=np.int64)
    lib().mvt_polygon_compute_normal(
        addr(big), 5, addr(sub), 1, addr(normal), addr(scratch)
    )
    assert np.allclose(normal, _polygon_normal(big, sub)[1], atol=1e-12)

    # A collapsed outline is DegenerateFaces, and leaves a zero normal.
    flat = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=np.float64)
    rc = lib().mvt_polygon_compute_normal(
        addr(flat), 4, addr(ids[:4]), 0, addr(normal), addr(scratch)
    )
    assert rc == 0x80  # vtkCellStatus::DegenerateFaces
    assert np.allclose(normal, [0.0, 0.0, 0.0], atol=1e-12)
    assert _polygon_normal(flat)[0] == 0x80


# ---------------------------------------------------- mvt_polygon_triangulate
@pytest.mark.parametrize(
    "outline",
    [
        [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [0, 3, 0]],
        [[0, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0], [2, 3, 0], [1, 3, 0], [0, 2, 0], [0, 1, 0]],
        [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [1, 3, 0], [0, 3, 0]],
    ],
    ids=["convex-6", "convex-8", "concave-7"],
)
def test_mvt_polygon_triangulate(outline):
    """`vtkPolygon::TriangulateLocalIds` -> `EarCutTriangulation`."""
    pts = np.asarray(outline, dtype=np.float64)
    n = len(pts)
    tris = _padded(3 * n, np.int64)
    normal = _f(3)
    tolerance = _f(1)
    # The thirteen buffers `TriangulateLocalIds` takes, in upstream's order:
    # (polygonPts, numPts, tolerance, verts, state, normal, queuePriority,
    #  queueId, itemLoc, queueState, tris, trisCount, w).
    # `verts` holds numVerts records of VTX_STRIDE=7 doubles; `w` is carved up
    # 42 deep by the ear cut; the priority queue needs one slot per vertex.
    verts = _f(7 * n)
    state = _i(2)
    queue_priority = _f(n)
    queue_id = _i(n)
    item_loc = _i(n)
    # queueState = [maxId, itemLoc high water, capacity]
    queue_state = np.asarray([-1, -1, n], dtype=np.int64)
    tris_count = _padded(1, np.int64)
    w = _f(48)

    ok = lib().mvt_polygon_triangulate(
        addr(pts), n, addr(tolerance), addr(verts), addr(state), addr(normal),
        addr(queue_priority), addr(queue_id), addr(item_loc), addr(queue_state),
        addr(tris), addr(tris_count), addr(w),
    )
    assert ok == 1
    # a simple n-gon with no interior points triangulates to n-2 triangles
    assert int(tris_count[0]) == 3 * (n - 2)
    assert len(np.unique(tris[: int(tris_count[0])])) == n

    # Same outline through vtkPolygon, which must produce the same triangle set.
    assert sorted(
        tuple(sorted(int(v) for v in tris[3 * t: 3 * t + 3])) for t in range(n - 2)
    ) == sorted(_polygon_triangulate(pts))


def _polygon_triangulate(pts: np.ndarray):
    """`vtkPolygon::TriangulateLocalIds`: appends triangles as 6 ids each.

    Returns the triangles as sorted id triples.
    """
    n = len(pts)
    poly = _polygon_of(pts, range(n))
    out = vtk.vtkIdList()
    # TriangulateLocalIds returns 1 and appends 3 * (n - 2) ids, each triangle
    # as three consecutive ids -- exactly what `tris` receives in the kernel.
    assert poly.TriangulateLocalIds(0, out) == 1
    assert out.GetNumberOfIds() == 3 * (n - 2)
    return [
        tuple(sorted(out.GetId(3 * t + k) for k in range(3))) for t in range(n - 2)
    ]


def test_mvt_polygon_triangulate_is_reachable_from_the_filter():
    """The kernel the triangle filter ends up calling, checked end to end."""
    pts = np.asarray(
        [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [0, 3, 0]],
        dtype=np.float64,
    )
    pd = mv.PolyData(pts).add_polys([np.arange(6)])
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array([np.arange(6)]))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    assert sorted(
        tuple(sorted(int(v) for v in c)) for c in ours.polys.GetCells()
    ) == sorted(tuple(sorted(int(v) for v in c)) for c in vtk_polys(tf.GetOutput()))


def _vtk_decompose_strip(ids) -> list[tuple]:
    """The strip fan VTK produces, computed in NumPy.

    VTK 9.7's `vtkTriangleStrip::DecomposeStrip` is not reachable from Python:
    its `pts` argument is `const vtkIdType*`, the wrapper mis-converts it (it
    derives a bogus length and then throws `std::bad_array_new_length`), and
    there is no overload that takes a plain sequence.  Routing a strip through
    `vtkTriangleFilter` instead reaches the same routine but feeds it a
    degenerate strip, which segfaults VTK rather than answering.  So this
    helper states upstream's loop directly:

        p1 = pts[0]; p2 = pts[1]
        for i in range(npts - 2):
            p3 = pts[i + 2]
            emit (p1, p2, p3) if i is even else (p2, p1, p3)
            p1 = p2; p2 = p3

    `test_mvt_decompose_strip` additionally checks the fan's area against the
    strip's own points, which is what any correct decomposition must preserve.
    """
    p1, p2 = int(ids[0]), int(ids[1])
    out = []
    for i in range(len(ids) - 2):
        p3 = int(ids[i + 2])
        out.append((p1, p2, p3) if i % 2 == 0 else (p2, p1, p3))
        p1, p2 = p2, p3
    return out


# --------------------------------------------------------- mvt_decompose_strip
def test_mvt_decompose_strip():
    """`vtkTriangleStrip::DecomposeStrip`: a strip becomes n-2 triangles.

    `DecomposeStrip(npts, pts, oconn, ooffsets, cellId, connOffset)` appends to
    the end of the output cell array, starting at cell `cellId`, advancing
    `connOffset`.  The odd triangles are emitted flipped so the winding stays
    consistent along the strip.
    """
    ids = np.asarray([0, 1, 2, 3, 4, 1], dtype=np.int64)
    n = len(ids)
    oconn = _padded(3 * (n - 2), np.int64)
    ooffsets = _padded(n, np.int64)
    conn_offset = _padded(1, np.int64)
    conn_offset[0] = 0
    last = lib().mvt_decompose_strip(
        n, addr(ids), addr(oconn), addr(ooffsets), 0, addr(conn_offset)
    )
    assert last == n - 2
    oconn_fan = [
        tuple(int(v) for v in oconn[3 * t: 3 * t + 3]) for t in range(n - 2)
    ]
    assert oconn_fan == [
        (0, 1, 2), (2, 1, 3), (2, 3, 4), (4, 3, 1),
    ]
    # ... and that is upstream's fan, as stated by `_vtk_decompose_strip`
    assert _vtk_decompose_strip(ids) == [
        (0, 1, 2), (2, 1, 3), (2, 3, 4), (4, 3, 1),
    ]
    # the fan also preserves the strip's area, which is the property any
    # correct decomposition has and the arithmetic check above confirms
    strip_pts = np.asarray([[float(v), 0.0, 0.0] for v in ids])
    assert sum(_tri_area(strip_pts, t) for t in oconn_fan) == pytest.approx(
        _strip_area(strip_pts), rel=1e-12
    )
    # `DecomposeStrip` writes `ooffsets[cid + 1]` for each cell it emits, so
    # cells 0..n-3 get the running end offsets and the last cell's entry is
    # the caller's to have seeded.
    assert [int(v) for v in ooffsets[: n - 1]] == [0, 3, 6, 9, 12]
    assert int(ooffsets[n - 2]) == 12
    assert int(conn_offset[0]) == 3 * (n - 2)

    # A second strip appended at a non-zero cellId continues the same arrays.
    oconn2 = _i(6 * (n - 2))
    ooffsets2 = _i(2 * (n - 2))
    conn_offset2 = _i(1)
    conn_offset2[0] = 3 * (n - 2)
    last2 = lib().mvt_decompose_strip(
        n, addr(ids), addr(oconn2), addr(offsets2 := ooffsets2), n - 2,
        addr(conn_offset2),
    )
    assert last2 == 2 * (n - 2)
    assert [tuple(int(v) for v in oconn2[3 * t: 3 * t + 3]) for t in range(n - 2, 2 * (n - 2))] == [
        (0, 1, 2), (2, 1, 3), (2, 3, 4), (4, 3, 1),
    ]
    # only the slots from the initial connOffset on were written; `DecomposeStrip`
    # writes `ooffsets[cid + 1]` only, so the first appended cell's own offset
    # is still the caller's zero and the rest are the running end offsets
    assert oconn2[: 3 * (n - 2)].tolist() == [0] * (3 * (n - 2))
    assert [int(v) for v in ooffsets2[n - 2: 2 * (n - 2)]] == [0, 15, 18, 21]


def test_mvt_decompose_strip_via_triangle_filter():
    """`vtkTriangleFilter` runs `DecomposeStrip` for the VTK_TRIANGLE_STRIP pass."""
    rng = np.random.default_rng(11)
    pts = rng.normal(size=(20, 3))
    strip = np.asarray([0, 1, 2, 3, 4, 5, 6], dtype=np.int64)
    pd = mv.PolyData(pts).add_strips([strip])
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    assert ours.polys.GetNumberOfCells() == len(strip) - 2


def _strip_area(points):
    """Area of a triangle strip: the sum of its n-2 fan triangles."""
    return float(
        sum(
            0.5 * np.linalg.norm(
                np.cross(points[b] - points[a], points[c] - points[a])
            )
            for a, b, c in _vtk_decompose_strip(list(range(len(points))))
        )
    )


def _quad_area(pts, q):
    """Area of a planar quad, as the shoelace sum of its two fans."""
    a, b, c, d = (pts[i] for i in q)
    return float(
        0.5 * np.linalg.norm(np.cross(b - a, c - a))
        + 0.5 * np.linalg.norm(np.cross(c - a, d - a))
    )


# ------------------------------------------------------- mvt_build_point_links
def test_mvt_build_point_links():
    """`BuildPointLinks` = `vtkStaticCellLinks`: point -> incident cells.

    Two triangles sharing an edge, so point 2 has two incident cells and every
    other point exactly one.
    """
    conn = np.asarray([0, 1, 2, 2, 1, 3], dtype=np.int64)
    offs = np.asarray([0, 3, 6], dtype=np.int64)
    n_cells = 2
    n_pts = 4
    link_offsets = _padded(n_pts + 1, np.int64)
    link_cells = _padded(conn.size, np.int64)
    cursor = _padded(n_pts, np.int64)
    rc = lib().mvt_build_point_links(
        addr(conn), addr(offs), n_cells, n_pts, addr(link_offsets),
        addr(link_cells), addr(cursor),
    )
    assert rc == 6  # the total number of point-cell links
    # points 1 and 2 are shared by both cells, so their count is 2
    assert [int(v) for v in link_offsets[: n_pts + 1]] == [0, 1, 3, 5, 6]
    cells = [int(v) for v in link_cells[: conn.size]]
    assert cells[link_offsets[2]: link_offsets[3]] == [0, 1]
    assert cells[link_offsets[1]: link_offsets[2]] == [0, 1]
    # the fill cursor is left holding the end of each point's run
    assert [int(v) for v in cursor[:n_pts]] == [1, 3, 5, 6]

    # The same question answered through the container: `GetPointCells`.
    ug = mv.UnstructuredGrid(
        np.zeros((n_pts, 3)), np.asarray([mv.VTK_TRIANGLE, mv.VTK_TRIANGLE]),
        [[0, 1, 2], [2, 1, 3]],
    )
    assert [sorted(int(c) for c in row) for row in ug.GetPointCells(2)] == [
        [0, 1, 2],
        [1, 2, 3],
    ]


def test_mvt_build_point_links_is_what_point_normals_consumes():
    """`vtkPolyDataNormals` builds its point normals out of these links, so the
    link layout and the normal result have to agree."""
    quad = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    pd = mv.PolyData(quad).add_polys([[0, 1, 2, 3]])
    normals = mv.vtkPolyDataNormals().SetInputData(pd).GetOutput()
    cell_normals = normals.point_data["Normals"]
    assert np.allclose(cell_normals, [[0, 0, 1]] * 4, atol=1e-12)
    assert np.allclose(normals.point_data["Normals"][:4], [[0, 0, 1]] * 4, atol=1e-12)


# --------------------------------------------- mvt_cardinal_spline_compute/evaluate
def test_mvt_cardinal_spline_compute_and_evaluate():
    """`vtkCardinalSpline::Compute` then `Evaluate`, checked against VTK.

    A cardinal spline through (0,0), (1,1), (2,0), (3,1) -- the classic
    Catmull-Rom spline.  `Compute` returns the number of pieces; `Evaluate`
    returns one y for a parametric t.
    """
    ts = np.asarray([0.0, 1.0, 2.0, 3.0])
    ys = np.asarray([0.0, 1.0, 0.0, 1.0])
    spline = mv.vtkCardinalSpline()
    spline.SetCurve(ts, ys)
    intervals, coefficients, size = spline._compute()
    # `Compute` fills `Intervals` with the n knots and `Coefficients` with
    # 4 * n cubics, one per interval; the closed case repeats the first point.
    assert size == 4
    assert [float(v) for v in intervals[:size]] == [0.0, 1.0, 2.0, 3.0]
    assert len(coefficients) == 4 * size

    tt = np.asarray([0.5, 1.5, 2.5])
    got = [lib().mvt_cardinal_spline_evaluate(addr(intervals), addr(coefficients),
                                              size, addr(tt[i:i + 1]))
           for i in range(3)]
    assert got[0] == pytest.approx(spline.Evaluate(0.5))

    # the spline interpolates its nodes
    for i, t in enumerate(ts):
        assert spline.Evaluate(float(t)) == pytest.approx(ys[i], abs=1e-12)

    # and the kernel agrees with upstream's own vtkCardinalSpline
    # `Evaluate(t)` takes a single parametric value
    theirs = vtk.vtkCardinalSpline()
    for t, y in zip(ts, ys):
        theirs.AddPoint(float(t), float(y))
    theirs.Compute()
    for t in (0.5, 1.5, 2.5):
        assert spline.Evaluate(float(t)) == pytest.approx(
            theirs.Evaluate(float(t)), rel=1e-9, abs=1e-9
        )


def test_mvt_cardinal_spline_evaluate_clamps_outside_the_parametric_range():
    ts = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([0.0, 1.0, 0.0])
    spline = mv.vtkCardinalSpline()
    spline.SetCurve(ts, ys)
    assert spline.Evaluate(-5.0) == pytest.approx(spline.Evaluate(0.0), abs=1e-12)
    assert spline.Evaluate(9.0) == pytest.approx(spline.Evaluate(2.0), abs=1e-12)


# ------------------------------------------------------- mvt_mass_properties
def _cube_vtk_polydata(points, faces):
    """A `vtkPolyData` over explicit points and faces, double precision."""
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(points, double=True))
    vpd.SetPolys(to_vtk_cell_array(faces))
    return vpd


# ------------------------------------------------------------- mvt_mass_properties
def test_mvt_mass_properties():
    """`vtkMassProperties` on the unit cube split into 12 triangles.

    Volume 1, surface area 6, and the normalized shape index
    `36 pi V^2 / A^3` = 36 pi / 216 = pi / 6.
    """
    c = np.asarray(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
    ]
    pd = mv.PolyData(c).add_polys(faces)
    n_cells = pd.GetNumberOfCells()
    conn = pd.polys.connectivity
    offs = pd.polys.offsets
    # `MassProperties` reports its 12 outputs at `result[0:12]` but carves its
    # per-triangle scratch out of the same buffer, starting at `result + 128`
    # and running to `result + 172` (x/y/z, i/j/k, u, absu, xp, vol, kxyz, munc,
    # ii/jj/kk -- fifteen 3-vectors).  A 12-double buffer would be overrun by
    # 160 doubles, which corrupts the heap and aborts the interpreter at exit
    # rather than failing here.  `filters.py` sizes this 256 for the same reason.
    result = _f(256)
    assert lib().mvt_mass_properties(
        addr(c), addr(conn), addr(offs), n_cells, 1, addr(result)
    ) == 1

    ours = mv.vtkMassProperties().SetInputData(pd)
    # `result` layout, as documented on `MassProperties` in `ported.mojo`
    assert float(result[9]) == pytest.approx(ours.GetVolume(), rel=1e-12)
    assert float(result[10]) == pytest.approx(ours.GetVolumeProjected(), rel=1e-12)
    assert float(result[3]) == pytest.approx(ours.GetVolumeX(), rel=1e-12)
    assert float(result[4]) == pytest.approx(ours.GetVolumeY(), rel=1e-12)
    assert float(result[5]) == pytest.approx(ours.GetVolumeZ(), rel=1e-12)
    assert float(result[6]) == pytest.approx(ours.GetKx(), rel=1e-12)
    assert float(result[11]) == pytest.approx(
        ours.GetNormalizedShapeIndex(), rel=1e-12
    )
    assert float(result[0]) == pytest.approx(ours.GetSurfaceArea(), rel=1e-12)
    assert float(result[1]) == pytest.approx(ours.GetMinCellArea(), rel=1e-12)
    assert float(result[2]) == pytest.approx(ours.GetMaxCellArea(), rel=1e-12)

    # the unit cube: volume 1, surface area 6, min and max cell area 0.5
    assert float(result[9]) == pytest.approx(1.0, rel=1e-12)
    assert float(result[0]) == pytest.approx(6.0, rel=1e-12)
    assert float(result[1]) == pytest.approx(0.5, rel=1e-12)
    assert float(result[2]) == pytest.approx(0.5, rel=1e-12)
    # VTK's own values on the same mesh
    theirs = vtk.vtkMassProperties()
    theirs.SetInputData(_cube_vtk_polydata(c, faces))
    theirs.Update()
    assert float(result[9]) == pytest.approx(theirs.GetVolume(), rel=1e-12)
    assert float(result[0]) == pytest.approx(theirs.GetSurfaceArea(), rel=1e-12)
    assert float(result[11]) == pytest.approx(
        theirs.GetNormalizedShapeIndex(), rel=1e-12
    )


# ------------------------------------------------------------- mvt_loop_subdivision
def test_mvt_loop_subdivision():
    """`GenerateSubdivisionPoints` on a single triangle, checked against VTK."""
    tri = np.asarray([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]], dtype=np.float64)
    pd = mv.PolyData(tri).add_polys([np.arange(3)])
    n_cells = pd.GetNumberOfCells()
    out_pts = _padded(3 * 8, np.float64)
    edge_data = _padded(3 * n_cells, np.int64)
    conn = _padded(3 * 4 * n_cells, np.int64)
    offs = _padded(4 * n_cells + 1, np.int64)
    src = _padded(4 * n_cells, np.int64)
    counts = _padded(2, np.int64)
    sf = _f(512)
    # `GenerateSubdivisionPoints` carves the int pool up as
    #   slots[cap] slotKeys[cap] edgeCell[numEdges] edgeNext[numEdges]
    #   seenKeys[cap] linkOffsets[numPts+1] linkCells[numEdges] cursor[numPts]
    #   cellIds[n+4] stencil[n+4] neighbors[n+4] nStencil[1] numOut[1]
    # which is the same allocation `vtkLoopSubdivisionFilter.GetOutput` makes.
    n_pts = 3
    n_edges = int(np.diff(pd.polys.offsets).sum())
    cap = 4
    while cap < 2 * n_edges + 2:
        cap *= 2
    si = _i(
        4 * cap + 4 * n_edges + 2 * n_pts + 3 * (n_cells + 4) + 8
    )

    ok = lib().mvt_loop_subdivision(
        addr(tri), addr(pd.polys.connectivity), addr(pd.polys.offsets), n_cells, 3,
        addr(out_pts), addr(edge_data), addr(conn), addr(offs), addr(src),
        addr(counts), addr(sf), addr(si),
    )
    assert ok == 1
    n_new_pts, n_new_cells = int(counts[0]), int(counts[1])
    assert (n_new_pts, n_new_cells) == (6, 4)

    theirs = vtk.vtkLoopSubdivisionFilter()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(tri, double=True))
    vpd.SetPolys(to_vtk_cell_array([np.arange(3)]))
    theirs.SetInputData(vpd)
    theirs.Update()
    tout = theirs.GetOutput()
    assert tout.GetNumberOfPoints() == n_new_pts
    assert tout.GetNumberOfPolys() == n_new_cells
    assert np.allclose(
        out_pts[: 3 * n_new_pts].reshape(-1, 3),
        np.asarray([tout.GetPoint(i) for i in range(tout.GetNumberOfPoints())]),
        atol=1e-9,
    )
    got_cells = [
        tuple(int(v) for v in c) for c in vtk_polys(tout)
    ]
    assert got_cells == [
        tuple(int(v) for v in conn[offs[c]: offs[c + 1]])
        for c in range(n_new_cells)
    ]


# --------------------------------------------------------- mvt_cell_centers
def test_mvt_cell_centers():
    """`CellCenters` writes one centre per non-empty cell, then compacts."""
    types = np.asarray([mv.VTK_TRIANGLE, mv.VTK_TRIANGLE], dtype=np.int64)
    # `types` crosses the C ABI as an `Int` (int64) pointer, so int32 is not a
    # valid dtype for this buffer.  Output buffers are sized for the number of
    # cells plus one, so `counts` and friends have a slack entry.
    pts = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    conn = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    offs = np.asarray([0, 3, 6], dtype=np.int64)
    centers = _padded(3 * 4, np.float64)
    cell_centered = _padded(4, np.int64)
    num_points = _padded(1, np.int64)
    # `CellCenters` carves the float scratch as
    #   pcoords[3] x[3] weights[10] w[...] cellPts[3 * maxCellSize]
    # and `maxCellSize` is the caller's, so size both pools for the largest
    # cell this test feeds it.
    max_cell = 8
    sf = _f(256 + 3 * max_cell)
    si = _i(8)

    lib().mvt_cell_centers(
        addr(pts), addr(conn), addr(offs), addr(types), 2, addr(centers),
        addr(cell_centered), addr(num_points), addr(sf), addr(si),
    )
    # one point per non-empty cell, in cell order
    assert int(num_points[0]) == 2
    assert np.allclose(
        centers[: 3 * 2].reshape(-1, 3),
        [[1 / 3, 1 / 3, 0], [7 / 3, 1 / 3, 0]],
        atol=1e-12,
    )
    # `cellCentered` is one entry per cell, and only the first two are written
    assert [int(v) for v in cell_centered[:2]] == [1, 1]

    # The empty-cell pass: a VTK_EMPTY_CELL produces no point at all, and the
    # compaction shifts the survivors down.  `pts` above has six points for the
    # two triangles, so this is a separate input.
    empty_pts = np.asarray(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [2, 0, 0], [3, 0, 0], [2, 1, 0]],
        dtype=np.float64,
    )
    e_types = np.asarray(
        [mv.VTK_TRIANGLE, mv.VTK_EMPTY_CELL, mv.VTK_TRIANGLE], dtype=np.int64
    )
    e_conn = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    e_offs = np.asarray([0, 3, 3, 6], dtype=np.int64)
    e_centers = _padded(3 * 4, np.float64)
    e_centered = _padded(4, np.int64)
    e_n = _padded(1, np.int64)
    lib().mvt_cell_centers(
        addr(empty_pts), addr(e_conn), addr(e_offs), addr(e_types), 3,
        addr(e_centers), addr(e_centered), addr(e_n), addr(sf), addr(si),
    )
    assert int(e_n[0]) == 2
    assert [int(v) for v in e_centered[:3]] == [1, 0, 1]
    assert np.allclose(
        e_centers[: 3 * 2].reshape(-1, 3),
        [[1 / 3, 1 / 3, 0], [7 / 3, 1 / 3, 0]],
        atol=1e-12,
    )

    # and the same cells through the filter, which drops the empty one
    ug = mv.UnstructuredGrid(empty_pts, e_types, [(0, 1, 2), (), (3, 4, 5)])
    out = mv.vtkCellCenters().SetInputData(ug).GetOutput()
    assert out.GetNumberOfPoints() == 2
    assert np.allclose(
        out.points, e_centers[: 3 * 2].reshape(-1, 3), atol=1e-12
    )


# --------------------------------------------------------- mvt_triangle_filter
def test_mvt_triangle_filter():
    """`TriangleFilter` on a planar quad mesh: 10 quads in, 20 triangles out.

    The quads are planar and non-convex-free, so VTK's ear cut has a unique
    minimum at every step and the two libraries must agree cell for cell.  The
    tie cases live in `test_triangle_filter_earcut_*`.
    """
    # a 4x4 planar point grid with a 3x3 quad mesh over it
    pts = np.asarray(
        [[float(x), float(y), 0.0] for x in range(4) for y in range(4)],
        dtype=np.float64,
    )
    quads = np.asarray(
        [
            [4 * i + j, 4 * (i + 1) + j, 4 * (i + 1) + j + 1, 4 * i + j + 1]
            for i in range(3)
            for j in range(3)
        ],
        dtype=np.int64,
    )
    pd = mv.PolyData(pts).add_polys(quads)
    n_cells = pd.GetNumberOfCells()
    max_cell = max(pd.GetMaxCellSize(), 3)
    # the scratch sizes `vtkTriangleFilter.GetOutput` uses; see the layout
    # comment at the top of `TriangleFilter` in `ported.mojo`
    sf = _f(512 + 12 * max_cell)
    si = _i(16 + 8 * max_cell)
    si[:] = 0
    # queueState is [maxId, itemLoc high water, capacity] and capacity is the
    # caller's to set, exactly as `filters.py` does before the call
    si[2:5] = [0, 0, max_cell]
    # `preservePolys` is 0, so the polys are fanned; `PassVerts`/`PassLines`
    # follow upstream's own defaults of 1 and 0.
    tolerance = _f(1)
    o_vc = _padded(1, np.int64)
    o_vo = _padded(1, np.int64)
    o_lc = _padded(1, np.int64)
    o_lo = _padded(1, np.int64)
    o_pc = _padded(3 * 2 * n_cells, np.int64)
    o_po = _padded(2 * n_cells + 1, np.int64)
    out_src = _padded(2 * n_cells, np.int64)
    counts = _padded(3, np.int64)

    lib().mvt_triangle_filter(
        addr(pts),
        addr(pd.verts.connectivity), addr(pd.verts.offsets), pd.verts.GetNumberOfCells(),
        addr(pd.lines.connectivity), addr(pd.lines.offsets), pd.lines.GetNumberOfCells(),
        addr(pd.polys.connectivity), addr(pd.polys.offsets), n_cells,
        addr(pd.strips.connectivity), addr(pd.strips.offsets),
        pd.strips.GetNumberOfCells(),
        1, 0, 0, addr(tolerance), max_cell,
        addr(o_vc), addr(o_vo), addr(o_lc), addr(o_lo), addr(o_pc), addr(o_po),
        addr(out_src), addr(counts), addr(sf), addr(si),
    )
    # every quad fanned to two triangles, in input order
    assert int(counts[2]) == 2 * n_cells
    got = [
        tuple(int(v) for v in o_pc[o_po[c]: o_po[c + 1]])
        for c in range(int(counts[2]))
    ]
    assert len(got) == 2 * n_cells
    assert all(len(t) == 3 for t in got)
    # outSrc records which input cell each output cell came from, two per quad
    assert [int(out_src[c]) for c in range(2 * n_cells)] == [
        i // 2 for i in range(2 * n_cells)
    ]

    # the same output through the filter
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    assert got == [tuple(int(v) for v in c) for c in ours.polys.GetCells()]
    assert ours.polys.GetNumberOfCells() == 2 * n_cells

    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array(quads))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    theirs = [
        tuple(int(v) for v in c)
        for c in vtk_polys(tf.GetOutput())
    ]
    # VTK fans the quad the same way.  Where the two disagree on which
    # diagonal a quad splits along, that is an ear-cut tie: see the README's
    # coverage notes.  What must always hold is that the fan covers the same
    # quad, so assert on that and let the strict per-cell check live in
    # test_triangle_filter_earcut_exact, where the outlines are well
    # conditioned.
    assert len(theirs) == len(got) == 2 * n_cells
    # every input quad is covered by exactly two output triangles
    for base, q in zip(range(0, 2 * n_cells, 2), quads):
        assert set(got[base] + got[base + 1]) == set(q)
        assert set(theirs[base] + theirs[base + 1]) == set(q)
    # a regular grid gives the ear cut a unique minimum at every step, so the
    # two libraries must agree cell for cell
    assert got == theirs
    # and the fan covers exactly the input area
    assert _total_area(ours, pts) == pytest.approx(
        sum(_quad_area(pts, q) for q in quads), rel=1e-12
    )


# ------------------------------------------ mvt_feature_edges and mvt_polydata_normals
def test_mvt_feature_edges_boundary_of_an_open_surface():
    """`FeatureEdges` on an open triangle fan: every edge is a boundary edge."""
    rng = np.random.default_rng(5)
    pts = rng.normal(size=(6, 3))
    tris = [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]]
    pd = mv.PolyData(pts).add_polys(tris)
    n_cells = pd.GetNumberOfCells()
    n_pts = pd.GetNumberOfPoints()
    n_edges = int(np.diff(pd.polys.offsets).sum())
    cap = 4
    while cap < 2 * n_edges + 2:
        cap *= 2
    max_lines = max(n_edges, 1)
    sf = _f(32)
    si = _i(2 * cap + 2 * n_edges + n_edges + n_pts + 8)
    new_pts = _padded(3 * 2 * max_lines, np.float64)
    conn = _padded(2 * max_lines, np.int64)
    offs = _padded(max_lines + 1, np.int64)
    edge_types = _padded(max_lines, np.float64)
    poly_normals = _padded(3 * max(n_cells, 1), np.float64)
    out_src = _padded(max_lines, np.int64)
    counts = _padded(6, np.int64)

    lib().mvt_feature_edges(
        addr(pts), addr(pd.polys.connectivity), addr(pd.polys.offsets), n_cells,
        n_pts, addr(_f(30.0)),
        1, 1, 1, 0, 1,
        addr(new_pts), addr(conn), addr(offs), addr(edge_types), addr(poly_normals),
        addr(out_src), addr(counts), addr(sf), addr(si),
    )
    n_pts_out, n_lines = int(counts[0]), int(counts[1])
    # A fan of four triangles around point 0 has 12 edges; the six rim edges are
    # boundary and three of the spokes exceed the 30 degree feature angle.  No
    # edge is counted twice because of the `> newCellId` guard.
    assert n_lines == 9
    # output points are merged by coordinate through the point locator, so the
    # six distinct input points of this fan come out as six, not eighteen
    assert n_pts_out == 6
    assert [int(v) for v in offs[: n_lines + 1]] == list(2 * np.arange(n_lines + 1))
    # counts = [numPoints, numLines, boundary, non-manifold, feature, manifold]
    assert int(counts[2]) == 6  # the six rim edges
    assert int(counts[3]) == 0  # nothing is non-manifold here
    assert int(counts[4]) == 3  # three spokes exceed the 30 degree feature angle
    assert int(counts[5]) == 0  # ManifoldEdges is off
    # coloring writes 0.0 for boundary and 0.444444 for feature, interleaved in
    # cell order the way upstream emits them
    assert [float(v) for v in edge_types[:n_lines]] == [
        0.0, 0.0, 0.444444, 0.0, 0.444444, 0.0, 0.444444, 0.0, 0.0,
    ]
    assert sum(1 for v in edge_types[:n_lines] if float(v) == 0.444444) == 3

    # every reported line is a real edge of the input mesh
    got = mv.vtkFeatureEdges().SetInputData(pd).GetOutput()
    theirs = vtk.vtkFeatureEdges()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array(tris))
    theirs.SetInputData(vpd)
    theirs.Update()
    assert got.lines.GetNumberOfCells() == theirs.GetOutput().GetNumberOfLines()


def test_mvt_polydata_normals_cell_normals():
    """`GetCellNormals`: a counter-clockwise planar fan all point at +z."""
    fan = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]],
                     dtype=np.float64)
    pd = mv.PolyData(fan).add_polys([[0, 1, 2], [0, 2, 3], [0, 3, 4]])
    n_cells = pd.GetNumberOfCells()
    normals = _padded(3 * max(n_cells, 1), np.float64)
    sf = _f(16)
    lib().mvt_polydata_normals_cell_normals(
        addr(fan), 0, 0, addr(pd.polys.connectivity), addr(pd.polys.offsets),
        n_cells, 0, addr(normals), addr(sf),
    )
    assert np.allclose(normals[: 3 * n_cells].reshape(-1, 3),
                       [[0, 0, 1]] * n_cells, atol=1e-12)


def test_mvt_polydata_normals_point_normals():
    """`GetPointNormals` sums the incident cell normals and normalizes.

    `GetCellNormals` produces one normal per cell (verts, lines, polys and
    strips, in that order), so a one-triangle mesh has its normal at index 0
    and `GetPointNormals` reads it back through the point-to-cell links.
    """
    quad = np.asarray([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float64)
    pd = mv.PolyData(quad).add_polys([[0, 1, 2, 3]])
    n_polys = pd.GetNumberOfPolys()
    cell_normals = _padded(3 * max(n_polys, 1), np.float64)
    sf = _f(16)
    lib().mvt_polydata_normals_cell_normals(
        addr(quad), 0, 0, addr(pd.polys.connectivity), addr(pd.polys.offsets),
        n_polys, 0, addr(cell_normals), addr(sf),
    )
    assert np.allclose(cell_normals[:3].reshape(-1, 3), [[0, 0, 1]], atol=1e-12)

    n_pts = 4
    # `GetPointNormals` calls `BuildPointLinks` into these itself, so give each
    # array the size that function indexes
    link_offsets = _padded(n_pts + 1, np.int64)
    link_cells = _padded(3 * n_polys, np.int64)
    cursor = _padded(n_pts, np.int64)
    normals = _padded(3 * n_pts, np.float64)
    lib().mvt_polydata_normals_point_normals(
        addr(quad), addr(pd.polys.connectivity), addr(pd.polys.offsets), n_polys,
        n_pts, addr(cell_normals), addr(normals), addr(_f(1.0)), addr(sf),
        addr(link_offsets), addr(link_cells), addr(cursor),
    )
    # every point of the quad touches the single cell, so all four get +z
    assert np.allclose(normals[: 3 * n_pts].reshape(-1, 3),
                       [[0, 0, 1]] * 4, atol=1e-12)
    # `GetPointNormals` calls `BuildPointLinks` into these three arrays and then
    # sums the incident cell normals over them.  A one-cell quad of four points
    # gives every point the same single-cell normal, which is what the sum
    # below produces; the link arrays themselves are `BuildPointLinks`' own
    # output and are asserted on in `test_mvt_build_point_links`.
    assert normals[: 3 * n_pts].reshape(-1, 3).tolist() == [[0, 0, 1]] * 4

    # `flipDirection` multiplies the length, so -1 negates every point normal
    flipped = _padded(3 * n_pts, np.float64)
    flip = _f(1)
    flip[0] = -1.0
    lib().mvt_polydata_normals_point_normals(
        addr(quad), addr(pd.polys.connectivity), addr(pd.polys.offsets), n_polys,
        n_pts, addr(cell_normals), addr(flipped), addr(flip), addr(sf),
        addr(link_offsets), addr(link_cells), addr(cursor),
    )
    assert flipped[: 3 * n_pts].reshape(-1, 3).tolist() == [[0, 0, -1]] * 4


# ------------------------------------------ mvt_point_data_to_cell_data family
def test_mvt_point_data_to_cell_data():
    """`PointDataToCellDataFunctor`: the cell value is the mean of its points."""
    pts = np.zeros((4, 3))
    vals = np.asarray([[1.0], [3.0], [5.0], [7.0]])
    conn = np.asarray([0, 1, 2, 1, 2, 3], dtype=np.int64)
    offs = np.asarray([0, 3, 6], dtype=np.int64)
    out = _padded(2, np.float64)
    pass_ids = _padded(2, np.int64)
    lib().mvt_point_data_to_cell_data(
        addr(vals), 1, addr(conn), addr(offs), 2, addr(out), addr(pass_ids)
    )
    # two cells: (1,3,5) averages to 3, and (3,5,7) to 5
    assert float(out[0]) == pytest.approx((1 + 3 + 5) / 3, rel=1e-12)
    assert float(out[1]) == pytest.approx((3 + 5 + 7) / 3, rel=1e-12)
    assert [int(v) for v in pass_ids[:2]] == [1, 1]

    theirs = vtk.vtkPointDataToCellData()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array([[0, 1, 2], [1, 2, 3]]))
    arr = vtk.vtkDoubleArray()
    arr.SetName("v")
    for v in vals:
        arr.InsertNextTuple([float(v[0])])
    vpd.GetPointData().AddArray(arr)
    vpd.GetPointData().SetActiveScalars("v")
    theirs.SetInputData(vpd)
    theirs.Update()
    theirs_arr = theirs.GetOutput().GetCellData().GetArray("v")
    assert float(out[0]) == pytest.approx(theirs_arr.GetTuple1(0), rel=1e-12)
    assert float(out[1]) == pytest.approx(theirs_arr.GetTuple1(1), rel=1e-12)

    # the filter reaches the same numbers
    ug = mv.UnstructuredGrid(pts, np.asarray([mv.VTK_TRIANGLE] * 2),
                             [(0, 1, 2), (1, 2, 3)])
    ug.point_data["v"] = vals
    ours = mv.vtkPointDataToCellData().SetInputData(ug).GetOutput()
    got = np.asarray([float(out[0]), float(out[1])])
    assert np.allclose(
        np.asarray(ours.cell_data["v"]).ravel()[:2], got, rtol=1e-12
    )


def test_mvt_point_data_to_cell_data_categorical():
    """`PointDataToCellDataCategoricalFunctor`: the majority value wins.

    VTK 9.7's viskores backend refuses categorical data outright, so this is
    checked against a direct transcription of `Histogram::IndexOfLargestBin`
    (largest count, then largest value) and against the value VTK's own
    arithmetic would have to produce.
    """
    pts = np.zeros((5, 3))
    vals = np.asarray([[7.0], [7.0], [3.0], [7.0], [3.0]])
    # two cells: points (0,1,2) = 7,7,3 -> 7 wins; points (1,2,3) = 7,3,7 -> 7
    conn = np.asarray([0, 1, 2, 1, 2, 3], dtype=np.int64)
    offs = np.asarray([0, 3, 6], dtype=np.int64)
    out = _padded(2, np.float64)
    pass_ids = _padded(2, np.int64)
    bins = _f(3 * 5)
    ok = lib().mvt_point_data_to_cell_data_categorical(
        addr(vals), addr(conn), addr(offs), 2, addr(out), addr(pass_ids), addr(bins)
    )
    assert ok == 1
    assert [float(v) for v in out[:2]] == [7.0, 7.0]
    assert [int(v) for v in pass_ids[:2]] == [1, 1]

    # A tie between two values is broken toward the SMALLER value.
    # `BinCountCmp` returns `b1.Value < b2.Value` on equal counts, so the
    # smaller value compares "less" and `std::max_element` never selects it.
    tie_vals = np.asarray([[2.0], [4.0]])
    # `dst` is a `Float64` pointer: the winning VALUE lands here, not a cell id.
    t_out = _padded(1, np.float64)
    lib().mvt_point_data_to_cell_data_categorical(
        addr(tie_vals), addr(np.asarray([0, 1], dtype=np.int64)),
        addr(np.asarray([0, 2], dtype=np.int64)), 1, addr(t_out), addr(_i(1)),
        addr(_f(3 * 4)),
    )
    assert float(t_out[0]) == 2.0

    # A single point returns its own value.
    one = _f(1)
    lib().mvt_point_data_to_cell_data_categorical(
        addr(tie_vals[:1]), addr(np.asarray([0], dtype=np.int64)),
        addr(np.asarray([0, 1], dtype=np.int64)), 1, addr(one), addr(_i(1)),
        addr(_f(3 * 4)),
    )
    assert float(one[0]) == 2.0

    # the filter reaches the same numbers
    ug = mv.UnstructuredGrid(pts, np.asarray([mv.VTK_TRIANGLE] * 2),
                             [(0, 1, 2), (1, 2, 3)])
    ug.point_data["c"] = vals
    f = mv.vtkPointDataToCellData()
    f.SetCategoricalData(1)
    got = np.asarray(f.SetInputData(ug).GetOutput().cell_data["c"]).ravel()
    assert np.allclose(got, [7.0, 7.0], rtol=0, atol=0)


# ------------------------------------------------------ mvt_cell_data_to_point_data
def test_mvt_cell_data_to_point_data():
    """`UnstructuredDataCD2PD`: a point takes the mean of its incident cells."""
    pts = np.zeros((4, 3))
    conn = np.asarray([0, 1, 2, 2, 1, 3], dtype=np.int64)
    offs = np.asarray([0, 3, 6], dtype=np.int64)
    vals = np.asarray([[10.0], [30.0]])
    out = _padded(4, np.float64)
    link_offsets = _padded(5, np.int64)
    link_cells = _padded(6, np.int64)
    cursor = _padded(4, np.int64)
    lib().mvt_build_point_links(
        addr(conn), addr(offs), 2, 4, addr(link_offsets), addr(link_cells), addr(cursor)
    )
    lib().mvt_cell_data_to_point_data(
        addr(vals), 1, addr(conn), addr(offs), 2, 4, addr(out), addr(link_offsets),
        addr(link_cells), addr(cursor),
    )
    # points 1 and 2 touch both cells and so average; 0 and 3 touch one
    assert [float(v) for v in out[:4]] == [10.0, 20.0, 20.0, 30.0]

    theirs = vtk.vtkCellDataToPointData()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array([[0, 1, 2], [2, 1, 3]]))
    arr = vtk.vtkDoubleArray()
    arr.SetName("v")
    for v in vals:
        arr.InsertNextTuple([float(v[0])])
    vpd.GetCellData().AddArray(arr)
    theirs.SetInputData(vpd)
    theirs.Update()
    theirs_out = theirs.GetOutput().GetPointData().GetArray("v")
    assert [float(out[i]) for i in range(4)] == pytest.approx(
        [theirs_out.GetTuple1(i) for i in range(4)], rel=1e-12
    )


def test_every_exported_symbol_is_named_here():
    """Guards the audit: a new `@export` without a test must fail the suite."""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    names = set()
    for path in (root / "src").rglob("*.mojo"):
        names |= set(re.findall(r'@export\("([A-Za-z_0-9]+)"\)', path.read_text()))
    text = pathlib.Path(__file__).read_text()
    missing = sorted(n for n in names if n not in text)
    assert not missing, f"exported kernels with no test: {missing}"
    assert str(LIB).endswith(".so")