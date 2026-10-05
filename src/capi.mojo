"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric.  That is the
only place the port diverges from VTK's structure: the kernel bodies in
`ported.mojo` keep upstream's logic, and this file is just the marshalling.

Callers own all memory, including scratch, so nothing here allocates and
nothing here can leak.
"""

from mojovtk.ported import (
    BuildPointLinks,
    CardinalSplineCompute,
    CardinalSplineEvaluate,
    CellCenters,
    CellDataToPointData,
    ComputeNormal,
    DecomposeStrip,
    FeatureEdges,
    GenerateSubdivisionPoints,
    GetCellNormals,
    GetPointNormals,
    MassProperties,
    PointDataToCellData,
    PointDataToCellDataCategorical,
    TriangulateLocalIds,
    TriangleFilter,
)

comptime PtrF = Pointer[Float64, AnyOrigin[mut=True]]
comptime PtrI = Pointer[Int64, AnyOrigin[mut=True]]
comptime PtrT = Pointer[Int, AnyOrigin[mut=True]]


def p(addr: Int) -> PtrF:
    return PtrF(unsafe_from_address=addr)


def q(addr: Int) -> PtrI:
    return PtrI(unsafe_from_address=addr)


def t(addr: Int) -> PtrT:
    return PtrT(unsafe_from_address=addr)


# A scalar Float64 also crosses as an address, so the caller's buffer stays the
# only thing that is ever interpreted as data.
def fv(addr: Int) -> Float64:
    return p(addr)[0]


# ------------------------------------------------- Filters/Core
@export("mvt_triangle_filter")
def mvt_triangle_filter(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int, a7: Int, a8: Int,
    a9: Int, a10: Int, a11: Int, a12: Int, a13: Int, a14: Int, a15: Int, a16: Int,
    a17: Int, a18: Int, a19: Int, a20: Int, a21: Int, a22: Int, a23: Int, a24: Int,
    a25: Int, a26: Int, a27: Int,
) abi("C"):
    TriangleFilter(
        p(a0),
        q(a1),
        q(a2),
        a3,
        q(a4),
        q(a5),
        a6,
        q(a7),
        q(a8),
        a9,
        q(a10),
        q(a11),
        a12,
        a13 != 0,
        a14 != 0,
        a15 != 0,
        fv(a16),
        a17,
        q(a18),
        q(a19),
        q(a20),
        q(a21),
        q(a22),
        q(a23),
        q(a24),
        q(a25),
        p(a26),
        q(a27),
    )


@export("mvt_cell_centers")
def mvt_cell_centers(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int, a7: Int, a8: Int, a9: Int
) abi("C"):
    CellCenters(p(a0), q(a1), q(a2), t(a3), a4, p(a5), t(a6), q(a7), p(a8), q(a9))


@export("mvt_mass_properties")
def mvt_mass_properties(a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int) abi("C") -> Int:
    MassProperties(p(a0), q(a1), q(a2), a3, a4, p(a5))
    return 1


@export("mvt_feature_edges")
def mvt_feature_edges(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    a8: Int,
    a9: Int,
    a10: Int,
    a11: Int,
    a12: Int,
    a13: Int,
    a14: Int,
    a15: Int,
    a16: Int,
    a17: Int,
    a18: Int,
    a19: Int,
) abi("C"):
    FeatureEdges(
        p(a0),
        q(a1),
        q(a2),
        a3,
        a4,
        fv(a5),
        a6 != 0,
        a7 != 0,
        a8 != 0,
        a9 != 0,
        a10 != 0,
        p(a11),
        q(a12),
        q(a13),
        p(a14),
        p(a15),
        q(a16),
        q(a17),
        p(a18),
        q(a19),
    )


@export("mvt_polydata_normals_cell_normals")
def mvt_polydata_normals_cell_normals(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int, a7: Int, a8: Int
) abi("C"):
    GetCellNormals(p(a0), a1, a2, q(a3), q(a4), a5, a6, p(a7), p(a8))


@export("mvt_polydata_normals_point_normals")
def mvt_polydata_normals_point_normals(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    a8: Int,
    a9: Int,
    a10: Int,
) abi("C"):
    GetPointNormals(
        p(a0),
        q(a1),
        q(a2),
        a3,
        a4,
        p(a5),
        p(a6),
        fv(a7),
        q(a8),
        q(a9),
        q(a10),
    )


@export("mvt_point_data_to_cell_data")
def mvt_point_data_to_cell_data(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int
) abi("C"):
    PointDataToCellData(p(a0), a1, q(a2), q(a3), a4, p(a5), t(a6))


@export("mvt_point_data_to_cell_data_categorical")
def mvt_point_data_to_cell_data_categorical(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int
) abi("C") -> Int:
    return 1 if PointDataToCellDataCategorical(
        p(a0), q(a1), q(a2), a3, p(a4), t(a5), p(a6)
    ) else 0


@export("mvt_cell_data_to_point_data")
def mvt_cell_data_to_point_data(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int, a7: Int, a8: Int, a9: Int
) abi("C"):
    CellDataToPointData(p(a0), a1, q(a2), q(a3), a4, a5, p(a6), q(a7), q(a8), q(a9))


# ------------------------------------------- Common/DataModel
@export("mvt_polygon_compute_normal")
def mvt_polygon_compute_normal(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int
) abi("C") -> Int:
    return ComputeNormal(p(a0), a1, q(a2), a3, p(a4), p(a5))


@export("mvt_polygon_triangulate")
def mvt_polygon_triangulate(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int, a7: Int, a8: Int,
    a9: Int, a10: Int, a11: Int, a12: Int,
) abi("C") -> Int:
    # `TriangulateLocalIds(polygonPts, numPts, tolerance, verts, state, normal,
    #  queuePriority, queueId, itemLoc, queueState, tris, trisCount, w)`
    return 1 if TriangulateLocalIds(
        p(a0),
        a1,
        fv(a2),
        p(a3),
        q(a4),
        p(a5),
        p(a6),
        q(a7),
        q(a8),
        q(a9),
        q(a10),
        q(a11),
        p(a12),
    ) else 0


@export("mvt_decompose_strip")
def mvt_decompose_strip(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int
) abi("C") -> Int:
    return DecomposeStrip(a0, q(a1), q(a2), q(a3), a4, q(a5))


@export("mvt_build_point_links")
def mvt_build_point_links(
    a0: Int, a1: Int, a2: Int, a3: Int, a4: Int, a5: Int, a6: Int
) abi("C") -> Int:
    return BuildPointLinks(q(a0), q(a1), a2, a3, q(a4), q(a5), q(a6))


# --------------------------- Common/ComputationalGeometry
@export("mvt_cardinal_spline_compute")
def mvt_cardinal_spline_compute(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    a8: Int,
    a9: Int,
    a10: Int,
    a11: Int,
    a12: Int,
) abi("C") -> Int:
    return CardinalSplineCompute(
        p(a0),
        p(a1),
        a2,
        a3 != 0,
        p(a4),
        p(a5),
        p(a6),
        p(a7),
        p(a8),
        a9,
        fv(a10),
        a11,
        fv(a12),
    )


@export("mvt_cardinal_spline_evaluate")
def mvt_cardinal_spline_evaluate(
    a0: Int, a1: Int, a2: Int, a3: Int
) abi("C") -> Float64:
    return CardinalSplineEvaluate(p(a0), p(a1), a2, fv(a3))


# -------------------------------------------- Filters/Modeling
@export("mvt_loop_subdivision")
def mvt_loop_subdivision(
    a0: Int,
    a1: Int,
    a2: Int,
    a3: Int,
    a4: Int,
    a5: Int,
    a6: Int,
    a7: Int,
    a8: Int,
    a9: Int,
    a10: Int,
    a11: Int,
    a12: Int,
) abi("C") -> Int:
    return GenerateSubdivisionPoints(
        p(a0),
        q(a1),
        q(a2),
        a3,
        a4,
        p(a5),
        q(a6),
        q(a7),
        q(a8),
        q(a9),
        q(a10),
        p(a11),
        q(a12),
    )

