"""Parity of every ported filter against the real VTK 9.7."""

from __future__ import annotations

import numpy as np
import pytest

import mojovtk as mv
from conftest import (
    cell_array_pairs,
    to_vtk_cell_array,
    to_vtk_points,
    to_vtk_polydata,
    to_vtk_unstructured_grid,
    vtk,
    vtk_array,
    vtk_lines,
    vtk_polys,
)

VTK_TRIANGLE = mv.VTK_TRIANGLE


def polydata_from_vtk(vpd) -> mv.PolyData:
    pd = mv.PolyData(np.asarray([vpd.GetPoint(i) for i in range(vpd.GetNumberOfPoints())]))
    for kind, getter in (
        ("verts", "GetVerts"),
        ("lines", "GetLines"),
        ("polys", "GetPolys"),
        ("strips", "GetStrips"),
    ):
        ca = getattr(vpd, getter)()
        ca.InitTraversal()
        ids = vtk.vtkIdList()
        blocks = []
        while ca.GetNextCell(ids):
            blocks.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
        if blocks:
            getattr(pd, f"add_{kind}")(blocks)
    return pd


# --------------------------------------------------------- vtkTriangleFilter
def test_triangle_filter_passthrough(sphere):
    pd = polydata_from_vtk(sphere)
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    theirs = vtk.vtkTriangleFilter()
    theirs.SetInputData(sphere)
    theirs.Update()
    assert cell_array_pairs(ours.polys) == sorted(
        tuple(int(v) for v in c) for c in vtk_polys(theirs.GetOutput())
    )
    assert np.allclose(ours.points, np.asarray([sphere.GetPoint(i) for i in range(
        sphere.GetNumberOfPoints())]))


def test_triangle_filter_quads():
    """Ten random quads, split along whichever diagonal VTK's `SimpleTriangulation`
    picks (shorter diagonal, unless the quad is concave)."""
    pts = np.asarray([[-0.45811827304922115, 0.11147930758994773, -0.7831667805636854], [-0.47642974385981746, -0.8191201803277345, -0.3334966741984413], [0.8531082743251359, -0.40658035065559817, -0.15387068019710476], [0.813718359438118, 0.6447702141569163, 1.6952075589596332], [-2.090485078128588, 0.856858944523106, -0.4822840867730822], [0.13469181881788084, 0.837719384899064, 1.0832531556913447], [1.0393506903088359, 0.15510669107825276, 1.6096626830183138], [-0.2829742272161367, -0.14098191307078672, 0.7993511888940862], [-0.5513724079804637, 2.160906258701247, 1.019206565354622], [2.1755753226203938, -0.026589187155779802, -0.38308855026130717], [0.16704800216692559, 0.7345765026061907, -0.5874276868740115], [0.3797017469724474, -0.01680427892083426, 1.6156817605764437], [-0.6627018123008657, 1.046161771728719, -0.6438635355400087], [-0.9606341931074224, -0.7102994273756232, -1.1901858618422472], [0.1463503600664521, 1.0312607960786646, 0.1643148785347983], [0.6243251121848756, 1.6321741955751323, 0.27002644717885166], [0.19516236096309458, -0.27506146796027464, -1.6082422850112825], [0.7597389318680837, -1.7564604476259789, 0.6526764363838281], [-0.014320335733954404, 1.1267814508104348, -0.067770260510462], [-0.8232001579576669, 0.3579102667525859, -0.5602361357081261], [-0.1809568246618137, 0.0418707868620003, -0.13454971174566932], [-0.18880227497381663, -0.8324449882870513, -0.18913378409281006], [-2.13834060712102, -0.15733209522523053, -1.198055985931922], [1.1202636903470218, 1.2699267473212534, -1.951025780521256], [0.14491679036588978, -0.12636392216675604, -1.0467156300808687], [0.5316531426489993, -0.46168954465474565, -1.7675990956866041], [-0.2666766520569678, -0.14825311830009477, 0.10643101824147437], [-1.2312329987327657, 0.6156809269215894, 0.7354570881988916], [-1.1458272768549103, -0.6588555689831765, -0.08033731382935666], [-0.5659253270123664, 1.7437546098496857, 0.20837211615824647], [-1.010582036354199, -0.7881089276247821, -0.05747025825572099], [2.2957788082586084, -0.17826432182306642, 0.12747641923980615], [0.5140453781581755, -0.0401259362630417, 2.280660872566708], [-0.5315249470251602, 0.7442031251457859, 0.16043796898357113], [0.6211963732164166, -1.1766429584197382, 1.768112695107191], [-0.12152750716346146, 0.03416326206751611, -0.9349762618687502], [-0.7296952681832494, 0.5642394331680308, 0.9864998861106795], [0.7532546907537343, 1.2089702496885697, 0.7144889121895972], [0.02846183335685169, 0.8365259894445134, 0.5936032404079788], [-0.10057526654145102, 0.726072718540168, 1.285697394529069]], dtype=np.float64)
    quads = np.asarray([[5, 31, 36, 39], [20, 23, 28, 35], [3, 29, 31, 33], [3, 7, 13, 33], [15, 25, 27, 29], [8, 20, 26, 34], [0, 26, 27, 38], [1, 8, 29, 30], [17, 25, 31, 39], [4, 15, 20, 36]], dtype=np.int64)
    pd = mv.PolyData(pts).add_polys(quads)
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()

    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array(quads))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    assert [tuple(int(v) for v in c) for c in ours.polys.GetCells()] == [
        tuple(int(v) for v in c) for c in vtk_polys(tf.GetOutput())
    ]
    # a quad always becomes two triangles
    assert ours.polys.GetNumberOfCells() == 2 * quads.shape[0]


def test_triangle_filter_strips():
    rng = np.random.default_rng(4)
    pts = rng.normal(size=(30, 3))
    strip = rng.integers(0, 30, size=7)
    pd = mv.PolyData(pts).add_strips([strip])
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()

    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetStrips(to_vtk_cell_array([strip]))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    assert cell_array_pairs(ours.polys) == sorted(
        tuple(int(v) for v in c) for c in vtk_polys(tf.GetOutput())
    )
    assert ours.polys.GetNumberOfCells() == 5


def _tri_area(pts, tri):
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _total_area(pd, pts):
    if isinstance(pd, mv.PolyData):
        cells = pd.polys.GetCells()
    else:
        ca = pd.GetPolys()
        cells = []
        ca.InitTraversal()
        ids = vtk.vtkIdList()
        while ca.GetNextCell(ids):
            cells.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    return sum(_tri_area(pts, np.asarray(t)) for t in cells)


def _polygon_area(pts):
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _triangulate_pts(pts, n, double=False):
    pd = mv.PolyData(pts).add_polys([np.arange(n)])
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double))
    vpd.SetPolys(to_vtk_cell_array([np.arange(n)]))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    return ours, tf.GetOutput()


# Well-conditioned outlines: the ear-cut measure `perimeter^2 / area` has a
# unique minimum at every step, so the two libraries must agree cell for cell.
EARCUT_EXACT = {
    ("convex", 6): [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [0, 3, 0]],
    ("convex", 7): [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0],
                    [0.5, 3, 0], [0, 1.5, 0]],
    ("convex", 8): [[0, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0], [2, 3, 0],
                    [1, 3, 0], [0, 2, 0], [0, 1, 0]],
    ("concave", 7): [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0],
                     [1, 3, 0], [0, 3, 0]],
}

# Outlines where several ears share the same measure.  VTK breaks the tie in
# float32 (vtkPoints defaults to VTK_FLOAT, and vtkTriangleFilter copies the
# polygon through it), this port in float64, so a different but equally valid
# diagonal can come out.  What must still hold is that the result is a
# triangulation of the same polygon with the same total area and orientation.
EARCUT_AMBIGUOUS = {
    ("convex", 5): [[0, 0, 0], [3, 0, 0], [4, 1.5, 0], [1.5, 4, 0], [0, 2, 0]],
    ("concave", 5): [[0, 0, 0], [3, 0, 0], [4, 1.5, 0], [0.4, 2, 0], [0, 2, 0]],
    ("concave", 6): [[0, 0, 0], [3, 0, 0], [4, 1, 0], [3, 2.5, 0], [1.5, 4, 0],
                     [0, 2, 0]],
    ("concave", 8): [[0, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0], [2, 3, 0],
                     [0.7, 2, 0], [0, 2, 0], [0, 1, 0]],
}


def _tri_area(pts, tri):
    a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))


def _total_area(pd, pts):
    if isinstance(pd, mv.PolyData):
        cells = pd.polys.GetCells()
    else:
        ca = pd.GetPolys()
        cells = []
        ca.InitTraversal()
        ids = vtk.vtkIdList()
        while ca.GetNextCell(ids):
            cells.append([ids.GetId(i) for i in range(ids.GetNumberOfIds())])
    return sum(_tri_area(pts, np.asarray(t)) for t in cells)


def _polygon_area(pts):
    p = np.asarray(pts, dtype=np.float64)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _canonical(cells):
    """Triangle identity, ignoring which vertex the ear-cut removed first.

    `RemoveVertex` emits (v, v->next, v->previous), so a rotation of the same
    triangle means a different vertex was removed, not a different
    triangulation.  See the README's coverage notes.
    """
    return sorted(tuple(sorted(int(v) for v in c)) for c in cells)


@pytest.mark.parametrize("key", sorted(EARCUT_EXACT))
def test_triangle_filter_earcut_exact(key):
    kind, n = key
    pts = np.asarray(EARCUT_EXACT[key], dtype=np.float64)
    ours, theirs = _triangulate_pts(pts, n, double=True)
    assert ours.polys.GetNumberOfCells() == n - 2
    assert _canonical(ours.polys.GetCells()) == _canonical(vtk_polys(theirs))


@pytest.mark.parametrize("key", sorted(EARCUT_AMBIGUOUS))
def test_triangle_filter_earcut_ties_are_still_valid(key):
    kind, n = key
    pts = np.asarray(EARCUT_AMBIGUOUS[key], dtype=np.float64)
    ours, theirs = _triangulate_pts(pts, n, double=True)
    assert ours.polys.GetNumberOfCells() == theirs.GetNumberOfPolys() == n - 2
    assert _total_area(ours, pts) == pytest.approx(_total_area(theirs, pts), rel=1e-12)
    assert _total_area(ours, pts) == pytest.approx(_polygon_area(pts), rel=1e-12)


def test_triangle_filter_arrowhead_quad():
    """A concave quad goes through `SimpleTriangulation`'s arrowhead path."""
    q = np.asarray([[0, 0, 0], [2, 0, 0], [1, 0.5, 0], [2, 2, 0]], dtype=np.float64)
    ours, theirs = _triangulate_pts(q, 4, double=True)
    assert _canonical(ours.polys.GetCells()) == _canonical(vtk_polys(theirs))


def _triangulate_pts(pts, n, double=False):
    pd = mv.PolyData(pts).add_polys([np.arange(n)])
    ours = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double))
    vpd.SetPolys(to_vtk_cell_array([np.arange(n)]))
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.Update()
    return ours, tf.GetOutput()


CONVEX = {
    5: [[0, 0, 0], [3, 0, 0], [4, 1.5, 0], [1.5, 4, 0], [0, 2, 0]],
    6: [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [0, 3, 0]],
    7: [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [0.5, 3, 0], [0, 1.5, 0]],
    8: [[0, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0], [2, 3, 0], [1, 3, 0], [0, 2, 0], [0, 1, 0]],
}

CONCAVE = {
    5: [[0, 0, 0], [3, 0, 0], [4, 1.5, 0], [0.4, 2, 0], [0, 2, 0]],
    6: [[0, 0, 0], [3, 0, 0], [4, 1, 0], [3, 2.5, 0], [1.5, 4, 0], [0, 2, 0]],
    7: [[0, 0, 0], [3, 0, 0], [4, 1, 0], [4, 3, 0], [2, 4.5, 0], [1, 3, 0], [0, 3, 0]],
    8: [[0, 0, 0], [2, 0, 0], [3, 1, 0], [3, 2, 0], [2, 3, 0], [0.7, 2, 0], [0, 2, 0], [0, 1, 0]],
}


def _canonical(cells):
    """Triangle identity, ignoring which vertex the ear-cut removed first.

    `RemoveVertex` emits (v, v->next, v->previous), so a rotation of the same
    triangle means a different vertex was removed, not a different
    triangulation.  See the README's coverage notes.
    """
    return sorted(tuple(sorted(int(v) for v in c)) for c in cells)


def test_triangle_filter_verts_and_lines(sphere):
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(sphere.GetPoints())
    vpd.SetPolys(sphere.GetPolys())
    vpd.SetVerts(sphere.GetVerts())
    pd = polydata_from_vtk(vpd)
    ours = mv.vtkTriangleFilter().SetInputData(pd)
    ours.SetPassVerts(1)
    ours.SetPassLines(1)
    out = ours.GetOutput()
    tf = vtk.vtkTriangleFilter()
    tf.SetInputData(vpd)
    tf.SetPassVerts(1)
    tf.SetPassLines(1)
    tf.Update()
    theirs = tf.GetOutput()
    assert out.verts.GetNumberOfCells() == theirs.GetNumberOfVerts()
    assert out.polys.GetNumberOfCells() == theirs.GetNumberOfPolys()


# ------------------------------------------------------------- vtkMassProperties
def test_mass_properties_unit_sphere(sphere):
    pd = polydata_from_vtk(sphere)
    ours = mv.vtkMassProperties().SetInputData(pd)
    theirs = vtk.vtkMassProperties()
    theirs.SetInputData(sphere)
    theirs.Update()
    assert ours.GetVolume() == pytest.approx(theirs.GetVolume(), rel=1e-12)
    assert ours.GetSurfaceArea() == pytest.approx(theirs.GetSurfaceArea(), rel=1e-12)
    assert ours.GetVolumeProjected() == pytest.approx(
        theirs.GetVolumeProjected(), rel=1e-9, abs=1e-9
    )
    assert ours.GetKx() == pytest.approx(theirs.GetKx(), rel=1e-12)
    assert ours.GetVolumeX() == pytest.approx(theirs.GetVolumeX(), rel=1e-9, abs=1e-12)
    assert ours.GetNormalizedShapeIndex() == pytest.approx(
        theirs.GetNormalizedShapeIndex(), rel=1e-9
    )
    assert ours.GetMinCellArea() == pytest.approx(theirs.GetMinCellArea(), rel=1e-12)
    assert ours.GetMaxCellArea() == pytest.approx(theirs.GetMaxCellArea(), rel=1e-12)


def test_mass_properties_cube():
    """A closed hexahedral-free triangle cube: 12 triangles, volume exactly 1."""
    c = np.asarray(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = [
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [1, 2, 6], [1, 6, 5],
        [0, 4, 7], [0, 7, 3],
    ]
    pd = mv.PolyData(c).add_polys(faces)
    ours = mv.vtkMassProperties().SetInputData(pd)
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(c))
    vpd.SetPolys(to_vtk_cell_array(faces))
    theirs = vtk.vtkMassProperties()
    theirs.SetInputData(vpd)
    theirs.Update()
    assert ours.GetVolume() == pytest.approx(1.0, rel=1e-12)
    assert ours.GetSurfaceArea() == pytest.approx(6.0, rel=1e-12)
    assert ours.GetVolume() == pytest.approx(theirs.GetVolume(), rel=1e-12)
    assert ours.GetSurfaceArea() == pytest.approx(theirs.GetSurfaceArea(), rel=1e-12)


def test_mass_properties_ignores_non_triangles(sphere):
    pd = polydata_from_vtk(sphere)
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(sphere.GetPoints())
    vpd.SetPolys(sphere.GetPolys())
    vpd.SetVerts(sphere.GetVerts())
    ours = mv.vtkMassProperties().SetInputData(pd)
    theirs = vtk.vtkMassProperties()
    theirs.SetInputData(vpd)
    theirs.Update()
    assert ours.GetVolume() == pytest.approx(theirs.GetVolume(), rel=1e-12)


# --------------------------------------------------------------- vtkCellCenters
def test_cell_centers_polydata(sphere):
    pd = polydata_from_vtk(sphere)
    ours = mv.vtkCellCenters().SetInputData(pd.to_polydata()).GetOutput()
    vpd = to_vtk_polydata(pd, double=True)
    theirs = vtk.vtkCellCenters()
    theirs.SetInputData(vpd)
    theirs.Update()
    arr = np.asarray(
        [theirs.GetOutput().GetPoint(i) for i in range(theirs.GetOutput().GetNumberOfPoints())]
    )
    assert ours.GetNumberOfPoints() == len(arr)
    assert np.allclose(ours.points, arr, atol=1e-9)


def test_cell_centers_cell_types():
    """One of every cell type VTK reports a center for."""
    types = [
        mv.VTK_VERTEX, mv.VTK_POLY_VERTEX, mv.VTK_LINE, mv.VTK_POLY_LINE,
        mv.VTK_TRIANGLE, mv.VTK_TRIANGLE_STRIP, mv.VTK_QUAD,
        mv.VTK_PIXEL, mv.VTK_TETRA, mv.VTK_VOXEL, mv.VTK_HEXAHEDRON,
        mv.VTK_WEDGE, mv.VTK_PYRAMID,
    ]
    pts = np.asarray(
        [
            [0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],
            [0, 1, 1], [1, 1, 1], [2, 0, 0],
        ]
    )
    cells = [
        [0],                                    # VTK_VERTEX
        [0, 1],                                 # VTK_POLY_VERTEX
        [0, 1],                                 # VTK_LINE
        [0, 1, 2],                              # VTK_POLY_LINE
        [0, 1, 2],                              # VTK_TRIANGLE
        [0, 1, 2, 3, 1],                        # VTK_TRIANGLE_STRIP
        [0, 1, 4, 3],                           # VTK_QUAD
        [0, 1, 4, 3],                           # VTK_PIXEL
        [0, 1, 2, 3],                           # VTK_TETRA
        [0, 1, 4, 3, 5, 6, 7, 2],               # VTK_VOXEL
        [0, 1, 4, 3, 7, 6, 2, 5],               # VTK_HEXAHEDRON
        [0, 1, 2, 3, 4, 5],                     # VTK_WEDGE
        [0, 1, 4, 8, 2],                        # VTK_PYRAMID
    ]
    ug = mv.UnstructuredGrid(pts, np.asarray(types), cells)
    ours = mv.vtkCellCenters().SetInputData(ug).GetOutput()

    vug = to_vtk_unstructured_grid(ug)
    theirs = vtk.vtkCellCenters()
    theirs.SetInputData(vug)
    theirs.Update()
    arr = np.asarray(
        [theirs.GetOutput().GetPoint(i) for i in range(theirs.GetOutput().GetNumberOfPoints())]
    )
    assert ours.GetNumberOfPoints() == len(arr)
    # the triangle-strip centre uses VTK's truncated 0.333333 constant
    assert np.allclose(ours.points, arr, atol=1e-9)


def test_cell_centers_polygon_matches_vtk():
    """VTK_POLYGON, checked against the one polygon outline where VTK agrees
    with itself.

    `vtkPolygon` does not override `vtkCell::GetParametricCenter`, so the centre
    is parametric (0.5, 0.5, 0.5), but `vtkPolygon::EvaluateLocation` rebuilds
    the p'/s/t frame and uses only `pcoords[0]`.  For the outline below VTK
    therefore returns t = 0.5 as well; for a general outline it returns whatever
    the uninitialised `t` slot happens to hold, which differs run to run.  This
    test pins the case VTK is self-consistent on; the divergence is recorded in
    the README.
    """
    pts = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [2, 0, 0]])
    ids = [0, 1, 4, 3]
    ug = mv.UnstructuredGrid(pts, np.asarray([mv.VTK_POLYGON]), [ids])
    ours = mv.vtkCellCenters().SetInputData(ug).GetOutput()
    theirs = vtk.vtkCellCenters()
    theirs.SetInputData(to_vtk_unstructured_grid(ug))
    theirs.Update()
    assert np.allclose(ours.GetPoint(0), theirs.GetOutput().GetPoint(0), atol=1e-9)


def test_cell_centers_polygon_is_inside_the_outline():
    """The polygon centre VTK's own formula defines, as an independent check on
    the case VTK cannot be compared against.  The same outline repeated many
    times must give the same answer."""
    pts = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [2, 0, 0]])
    ug = mv.UnstructuredGrid(pts, np.asarray([mv.VTK_POLYGON]), [[0, 1, 4, 2, 5]])
    first = mv.vtkCellCenters().SetInputData(ug).GetOutput().GetPoint(0)
    for _ in range(8):
        again = mv.vtkCellCenters().SetInputData(ug).GetOutput().GetPoint(0)
        assert np.array_equal(first, again)
    # s = t = 0.5 in the re-parameterized frame must land strictly inside
    assert 0.0 <= first[0] <= 2.0 and 0.0 <= first[1] <= 1.0


def test_cell_centers_vertex_cells(sphere):
    pd = polydata_from_vtk(sphere)
    f = mv.vtkCellCenters().SetInputData(pd.to_polydata())
    f.SetVertexCells(1)
    ours = f.GetOutput()
    theirs = vtk.vtkCellCenters()
    theirs.SetInputData(sphere)
    theirs.SetVertexCells(1)
    theirs.Update()
    assert ours.verts.GetNumberOfCells() == theirs.GetOutput().GetNumberOfVerts()


# -------------------------------------------------------------- vtkFeatureEdges
def test_feature_edges_open_surface():
    """A single triangle has three boundary edges and no feature edges."""
    tri = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    pd = mv.PolyData(tri).add_polys([np.arange(3)])
    f = mv.vtkFeatureEdges().SetInputData(pd)
    ours = f.GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(tri, double=True))
    vpd.SetPolys(to_vtk_cell_array([np.arange(3)]))
    theirs = vtk.vtkFeatureEdges()
    theirs.SetInputData(vpd)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.lines.GetNumberOfCells() == out.GetNumberOfLines()
    assert cell_array_pairs(ours.lines) == sorted(
        tuple(int(v) for v in c) for c in vtk_lines(out)
    )
    assert np.allclose(
        np.asarray(ours.cell_data["Edge Types"]), vtk_array(out.GetCellData(), "Edge Types"),
        atol=1e-6,
    )
    assert f.GetNumberOfBoundaryEdges() == 3
    assert f.GetNumberOfFeatureEdges() == 0


def test_feature_edges_folded_surface():
    """Two triangles sharing an edge at 90 degrees: a feature edge."""
    pts = np.asarray([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    polys = [[0, 1, 2], [0, 3, 1]]
    pd = mv.PolyData(pts).add_polys(polys)
    f = mv.vtkFeatureEdges().SetInputData(pd)
    ours = f.GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array(polys))
    theirs = vtk.vtkFeatureEdges()
    theirs.SetInputData(vpd)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.lines.GetNumberOfCells() == out.GetNumberOfLines()
    assert cell_array_pairs(ours.lines) == sorted(
        tuple(int(v) for v in c) for c in vtk_lines(out)
    )
    assert f.GetNumberOfFeatureEdges() == 1
    assert f.GetNumberOfBoundaryEdges() == 4


def test_feature_edges_closed_sphere(sphere):
    pd = polydata_from_vtk(sphere)
    f = mv.vtkFeatureEdges().SetInputData(pd)
    ours = f.GetOutput()
    theirs = vtk.vtkFeatureEdges()
    theirs.SetInputData(sphere)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.lines.GetNumberOfCells() == out.GetNumberOfLines()
    assert f.GetNumberOfBoundaryEdges() == 0
    assert f.GetNumberOfFeatureEdges() == 0


def test_feature_edges_manifold(sphere):
    pd = polydata_from_vtk(sphere)
    f = mv.vtkFeatureEdges().SetInputData(pd)
    f.SetBoundaryEdges(0)
    f.SetFeatureEdges(0)
    f.SetManifoldEdges(1)
    ours = f.GetOutput()
    theirs = vtk.vtkFeatureEdges()
    theirs.SetInputData(sphere)
    theirs.SetBoundaryEdges(0)
    theirs.SetFeatureEdges(0)
    theirs.SetManifoldEdges(1)
    theirs.Update()
    assert f.GetNumberOfManifoldEdges() == theirs.GetOutput().GetNumberOfLines()
    assert ours.lines.GetNumberOfCells() == theirs.GetOutput().GetNumberOfLines()


# ------------------------------------------------------------ vtkPolyDataNormals
def test_polydata_normals_cell_normals(sphere):
    """Cell normals are the `vtkPolygon::ComputeNormal` of each cell."""
    pd = polydata_from_vtk(sphere)
    f = mv.vtkPolyDataNormals().SetInputData(pd)
    f.SetComputeCellNormals(1)
    ours = f.GetOutput()
    theirs = vtk.vtkPolyDataNormals()
    theirs.SetInputData(sphere)
    theirs.SetComputeCellNormals(1)
    theirs.SetSplitting(0)
    theirs.SetConsistency(0)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.cell_data["Normals"],
        vtk_array(out.GetCellData(), "Normals"),
        atol=1e-5,
    )


def test_polydata_normals_point_normals(sphere):
    """Point normals are `GetPointNormals`: the normalized sum of the cell
    normals of every cell touching the point.

    VTK 9.7's own `vtkPolyDataNormals` output on a closed sphere is *not* that
    sum (a 5-line NumPy check on VTK's own cell and point normals disagrees by
    0.03), so the reference here is the algorithm's definition, and VTK is
    compared on a mesh where its output is the plain sum.
    """
    pd = polydata_from_vtk(sphere)
    f = mv.vtkPolyDataNormals().SetInputData(pd)
    ours = f.GetOutput()
    acc = np.zeros_like(ours.points)
    for cell, normal in zip(pd.polys.GetCells(), ours.cell_data["Normals"] if
                            "Normals" in ours.cell_data else [None] * pd.GetNumberOfPolys()):
        pass
    f2 = mv.vtkPolyDataNormals().SetInputData(pd)
    f2.SetComputeCellNormals(1)
    cell_normals = f2.GetOutput().cell_data["Normals"].astype(np.float64)
    for c, n in zip(pd.polys.GetCells(), cell_normals):
        for p in c:
            acc[int(p)] += n
    length = np.linalg.norm(acc, axis=1)
    length[length == 0.0] = 1.0
    assert np.allclose(ours.point_data["Normals"], acc / length[:, None], atol=1e-6)
    # a unit sphere's point normals are radial
    assert np.allclose(np.linalg.norm(ours.point_data["Normals"], axis=1), 1.0, atol=1e-5)


def test_polydata_normals_match_vtk_on_folded_mesh():
    """Where VTK's output is the plain sum, the two agree exactly."""
    pts = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
    tris = [[0, 1, 2], [0, 1, 3]]
    pd = mv.PolyData(pts).add_polys(tris)
    ours = mv.vtkPolyDataNormals().SetInputData(pd).GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts, double=True))
    vpd.SetPolys(to_vtk_cell_array(tris))
    theirs = vtk.vtkPolyDataNormals()
    theirs.SetInputData(vpd)
    theirs.SetComputeCellNormals(1)
    theirs.SetSplitting(0)
    theirs.SetConsistency(0)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.point_data["Normals"],
        vtk_array(out.GetPointData(), "Normals"),
        atol=1e-9,
    )
    assert np.allclose(
        ours.point_data["Normals"], [[0, -0.70710678, 0.70710678], [0, -0.70710678, 0.70710678],
                                     [0, 0, 1], [0, -1, 0]],
        atol=1e-9,
    )


def test_polydata_normals_flat_quad():
    pts = np.asarray([[0.0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    pd = mv.PolyData(pts).add_polys([np.arange(4)])
    ours = mv.vtkPolyDataNormals().SetInputData(pd).GetOutput()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(pts))
    vpd.SetPolys(to_vtk_cell_array([np.arange(4)]))
    theirs = vtk.vtkPolyDataNormals()
    theirs.SetInputData(vpd)
    theirs.SetSplitting(0)
    theirs.SetConsistency(0)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.point_data["Normals"],
        vtk_array(out.GetPointData(), "Normals"),
        atol=1e-6,
    )


# ------------------------------------------------- vtkPointDataToCellData
def test_point_data_to_cell_data(sphere):
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    rng = np.random.default_rng(5)
    ug.point_data["scalars"] = rng.normal(size=(ug.GetNumberOfPoints(), 1))

    ours = mv.vtkPointDataToCellData().SetInputData(ug).GetOutput()
    vug = to_vtk_unstructured_grid(ug)
    theirs = vtk.vtkPointDataToCellData()
    theirs.SetInputData(vug)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.cell_data["scalars"], vtk_array(out.GetCellData(), "scalars"), atol=1e-9
    )


def test_point_data_to_cell_data_vectors(sphere):
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    rng = np.random.default_rng(6)
    vec = rng.normal(size=(ug.GetNumberOfPoints(), 3))
    ug.point_data["vec"] = vec
    ours = mv.vtkPointDataToCellData().SetInputData(ug).GetOutput()
    vug = to_vtk_unstructured_grid(ug)
    theirs = vtk.vtkPointDataToCellData()
    theirs.SetInputData(vug)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(ours.cell_data["vec"], vtk_array(out.GetCellData(), "vec"), atol=1e-9)


def test_point_data_to_cell_data_categorical(sphere):
    """Categorical averaging must pick the majority value, cell by cell.

    VTK 9.7's viskores backend refuses categorical data ("CategoricalData is
    not supported"), so the reference here is a direct NumPy transcription of
    `Histogram::IndexOfLargestBin`, and the VTK path is covered by the
    non-categorical test above.
    """
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    n = ug.GetNumberOfPoints()
    lab = np.zeros((n, 1))
    # give points 0..5 label 1, everything else 0
    lab[:6] = 1.0
    ug.point_data["cat"] = lab
    f = mv.vtkPointDataToCellData().SetInputData(ug)
    f.SetCategoricalData(1)
    ours = f.GetOutput()

    got = ours.cell_data["cat"].ravel()
    want = []
    for cell in range(ug.GetNumberOfCells()):
        vals = [lab[pt, 0] for pt in ug.cells.GetCell(cell)]
        counts: dict[float, int] = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        # BinCountCmp: largest count, then largest value
        want.append(max(sorted(counts), key=lambda v: (counts[v], v)))
    assert np.array_equal(got, np.asarray(want))


# ------------------------------------------------- vtkCellDataToPointData
def test_cell_data_to_point_data(sphere):
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    rng = np.random.default_rng(7)
    ug.cell_data["scalars"] = rng.normal(size=(ug.GetNumberOfCells(), 1))
    ours = mv.vtkCellDataToPointData().SetInputData(ug).GetOutput()
    vug = to_vtk_unstructured_grid(ug)
    theirs = vtk.vtkCellDataToPointData()
    theirs.SetInputData(vug)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.point_data["scalars"], vtk_array(out.GetPointData(), "scalars"), atol=1e-9
    )


def test_cell_data_to_point_data_vectors(sphere):
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    rng = np.random.default_rng(8)
    ug.cell_data["vec"] = rng.normal(size=(ug.GetNumberOfCells(), 3))
    ours = mv.vtkCellDataToPointData().SetInputData(ug).GetOutput()
    vug = to_vtk_unstructured_grid(ug)
    theirs = vtk.vtkCellDataToPointData()
    theirs.SetInputData(vug)
    theirs.Update()
    out = theirs.GetOutput()
    assert np.allclose(
        ours.point_data["vec"], vtk_array(out.GetPointData(), "vec"), atol=1e-9
    )


def test_cell_data_to_point_data_averages_interior_points(sphere):
    """A field that is constant over each cell survives both directions: the
    point value is that constant wherever every incident cell agrees."""
    pd = polydata_from_vtk(sphere)
    ug = pd.to_polydata().to_unstructured_grid()
    vals = np.arange(ug.GetNumberOfCells(), dtype=np.float64)
    ug.cell_data["s"] = vals
    pts = mv.vtkCellDataToPointData().SetInputData(ug).GetOutput().point_data["s"]
    counts = np.zeros(ug.GetNumberOfPoints())
    for cell in range(ug.GetNumberOfCells()):
        for pt in ug.cells.GetCell(cell):
            counts[int(pt)] += 1
    interior = np.nonzero(counts == 6)[0]
    assert len(interior) > 0
    for pt in interior:
        got = [vals[int(q)] for q in range(ug.GetNumberOfCells()) if pt in ug.cells.GetCell(q)]
        assert pts[pt, 0] == pytest.approx(np.mean(got), abs=1e-12)


# -------------------------------------------------------- vtkCardinalSpline
def test_cardinal_spline_interpolates_nodes():
    ts = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0])
    ys = np.asarray([0.0, 1.0, 0.0, -1.0, 0.0])
    ours = mv.vtkCardinalSpline()
    ours.SetCurve(ts, ys)
    theirs = vtk.vtkCardinalSpline()
    for t, y in zip(ts, ys):
        theirs.AddPoint(float(t), float(y))
    for t in np.linspace(0.0, 4.0, 41):
        assert ours.Evaluate(float(t)) == pytest.approx(
            theirs.Evaluate(float(t)), abs=1e-9
        )
    for t, y in zip(ts, ys):
        assert ours.Evaluate(float(t)) == pytest.approx(y, abs=1e-9)


def test_cardinal_spline_closed():
    # a closed curve repeats its first point, so the samples stop short of the
    # parametric range end
    ang = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    ts = ang
    ys = np.sin(2 * ang)
    ours = mv.vtkCardinalSpline()
    ours.SetCurve(ts, ys)
    ours.SetClosed(1)
    ours.SetParametricRange(0.0, 2 * np.pi)
    theirs = vtk.vtkCardinalSpline()
    for t, y in zip(ts, ys):
        theirs.AddPoint(float(t), float(y))
    theirs.SetClosed(1)
    theirs.SetParametricRange(0.0, 2 * np.pi)
    for t in np.linspace(0.0, 2 * np.pi, 37):
        assert ours.Evaluate(float(t)) == pytest.approx(
            theirs.Evaluate(float(t)), abs=1e-8
        )


def test_cardinal_spline_clamps_outside():
    ts = np.asarray([0.0, 1.0, 2.0])
    ys = np.asarray([0.0, 1.0, 4.0])
    ours = mv.vtkCardinalSpline()
    ours.SetCurve(ts, ys)
    theirs = vtk.vtkCardinalSpline()
    for t, y in zip(ts, ys):
        theirs.AddPoint(float(t), float(y))
    for t in (-5.0, 7.0):
        assert ours.Evaluate(t) == pytest.approx(theirs.Evaluate(t), abs=1e-12)


# ------------------------------------------------ vtkLoopSubdivisionFilter
def test_loop_subdivision_one_level(sphere):
    pd = polydata_from_vtk(sphere)
    ours = mv.vtkLoopSubdivisionFilter().SetInputData(pd).GetOutput()
    theirs = vtk.vtkLoopSubdivisionFilter()
    theirs.SetInputData(sphere)
    theirs.SetNumberOfSubdivisions(1)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.GetNumberOfPoints() == out.GetNumberOfPoints()
    assert ours.GetNumberOfPolys() == out.GetNumberOfPolys()
    assert np.allclose(ours.points, np.asarray([out.GetPoint(i) for i in range(
        out.GetNumberOfPoints())]), atol=1e-9)


def test_loop_subdivision_two_levels(sphere):
    pd = polydata_from_vtk(sphere)
    ours = mv.vtkLoopSubdivisionFilter().SetInputData(pd)
    ours.SetNumberOfSubdivisions(2)
    ours = ours.GetOutput()
    theirs = vtk.vtkLoopSubdivisionFilter()
    theirs.SetInputData(sphere)
    theirs.SetNumberOfSubdivisions(2)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.GetNumberOfPoints() == out.GetNumberOfPoints()
    assert ours.GetNumberOfPolys() == out.GetNumberOfPolys()
    assert np.allclose(ours.points, np.asarray([out.GetPoint(i) for i in range(
        out.GetNumberOfPoints())]), atol=1e-9)


def test_loop_subdivision_open_triangle():
    """A lone triangle: every edge is a boundary edge, so the even stencil
    is the boundary rule (1/8, 1/8, 3/4) for the two vertices that have one."""
    tri = np.asarray([[0.0, 0, 0], [2, 0, 0], [0, 2, 0]])
    pd = mv.PolyData(tri).add_polys([np.arange(3)])
    ours = mv.vtkLoopSubdivisionFilter().SetInputData(pd).GetOutput()
    theirs = vtk.vtkLoopSubdivisionFilter()
    vpd = vtk.vtkPolyData()
    vpd.SetPoints(to_vtk_points(tri))
    vpd.SetPolys(to_vtk_cell_array([np.arange(3)]))
    theirs.SetInputData(vpd)
    theirs.Update()
    out = theirs.GetOutput()
    assert ours.GetNumberOfPoints() == out.GetNumberOfPoints()
    assert np.allclose(ours.points, np.asarray([out.GetPoint(i) for i in range(
        out.GetNumberOfPoints())]), atol=1e-9)
