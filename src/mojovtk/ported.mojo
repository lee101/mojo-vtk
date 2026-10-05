"""Port of VTK's unstructured-grid filter kernels to Mojo.

Every function below is transcribed from the VTK 9.7.0 source named in the
section header, keeping upstream's names, argument order, branch structure and
arithmetic, so this file reads top to bottom like the files it came from.

Memory convention, mirrored in `src/capi.mojo`: nothing here allocates, so
nothing here can leak.  Callers own every buffer including scratch.

* A "cell array" is the pair `(connectivity, offsets)` with
  `offsets[nCells] == len(connectivity)` -- exactly VTK's modern
  `vtkCellArray` layout.
* Points are a flat `(numPoints, 3)` float64 array.
* A `double x[3]` local becomes a three-double window of a caller-supplied
  scratch `PtrF`, so the transcription is literal rather than a rewrite.
  A `double* weights` becomes a window sized by the cell.

The one structural divergence is the link/edge-table machinery: upstream gets
`vtkPolyData::GetPointCells` and `GetCellEdgeNeighbors` from `vtkPolyData`'s
cached links and an octree, and those caches are built in C++ with types Mojo
has no equivalent of.  `BuildPointLinks` and `BuildEdgeTable` below build the
same two indices into caller-owned flat arrays, and every consumer asks them
exactly the question upstream asks its links.
"""

from std.math import abs, cbrt, cos, sin, sqrt
from std.utils import StaticTuple

comptime PtrF = Pointer[Float64, AnyOrigin[mut=True]]
comptime PtrI = Pointer[Int64, AnyOrigin[mut=True]]
comptime PtrT = Pointer[Int, AnyOrigin[mut=True]]

comptime VTK_DOUBLE_MAX: Float64 = 1.7976931348623157e308
comptime VTK_DOUBLE_MIN: Float64 = 2.2250738585072014e-308
comptime VTK_TOL = 1.0e-8
comptime VTK_PI = 3.14159265358979323846

# ===================================================================
# Common/Core/vtkMath.cxx
# ===================================================================


def Subtract(a: PtrF, b: PtrF, c: PtrF):
    c[0] = a[0] - b[0]
    c[1] = a[1] - b[1]
    c[2] = a[2] - b[2]


def Add(a: PtrF, b: PtrF, c: PtrF):
    c[0] = a[0] + b[0]
    c[1] = a[1] + b[1]
    c[2] = a[2] + b[2]


# The temporaries are upstream's: `Cross(v1, v2, v1)` is a legal call, so the
# inputs must be read before the output is written.
def Cross(a: PtrF, b: PtrF, c: PtrF):
    var cx = a[1] * b[2] - a[2] * b[1]
    var cy = a[2] * b[0] - a[0] * b[2]
    var cz = a[0] * b[1] - a[1] * b[0]
    c[0] = cx
    c[1] = cy
    c[2] = cz


def Dot(a: PtrF, b: PtrF) -> Float64:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def MultiplyScalar(a: PtrF, s: Float64):
    a[0] = a[0] * s
    a[1] = a[1] * s
    a[2] = a[2] * s


def Norm(a: PtrF) -> Float64:
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def SquaredNorm(a: PtrF) -> Float64:
    return a[0] * a[0] + a[1] * a[1] + a[2] * a[2]


def Normalize(a: PtrF) -> Float64:
    var n = Norm(a)
    if n != 0.0:
        a[0] = a[0] / n
        a[1] = a[1] / n
        a[2] = a[2] / n
    return n


def Distance2BetweenPoints(a: PtrF, b: PtrF) -> Float64:
    var d0 = a[0] - b[0]
    var d1 = a[1] - b[1]
    var d2 = a[2] - b[2]
    return d0 * d0 + d1 * d1 + d2 * d2


def RadiansFromDegrees(degrees: Float64) -> Float64:
    return degrees * VTK_PI / 180.0


# `SolveLinearSystem` dispatches a 2x2 system to this; `x` receives the pair.
def SolveLinearSystemGEPP2x2(
    a00: Float64,
    a01: Float64,
    a10: Float64,
    a11: Float64,
    b0: Float64,
    b1: Float64,
    x: PtrF,
) -> Bool:
    var c00 = a00
    var c01 = a01
    var c10 = a10
    var c11 = a11
    var d0 = b0
    var d1 = b1
    # Check if any of the matrix coefficients is zero.  If so then swap
    # rows/columns to form an upper triangular matrix without having to use
    # GEPP.
    var cols_swapped = False
    if (c00 == 0.0) or (c01 == 0.0) or (c10 == 0.0) or (c11 == 0.0):
        # zero in either row of the 2nd column?
        if (c01 == 0.0) or (c11 == 0.0):
            var t = c00
            c00 = c01
            c01 = t
            t = c10
            c10 = c11
            c11 = t
            cols_swapped = True
        # zero in a00?
        if c00 == 0.0:
            var t = c00
            c00 = c10
            c10 = t
            t = c01
            c01 = c11
            c11 = t
            t = d0
            d0 = d1
            d1 = t
    else:
        # None of the matrix coefficients are exactly zero.  Use GEPP to form
        # an upper triangular matrix, i.e. so that a10 == 0.  Select pivot by
        # looking at largest absolute value in a00, a10.
        if abs(c00) < abs(c10):
            var t = c00
            c00 = c10
            c10 = t
            t = c01
            c01 = c11
            c11 = t
            t = d0
            d0 = d1
            d1 = t
        var f = -c10 / c00
        c11 += c01 * f
        d1 += d0 * f
    # Have now an exact zero in a10.  Need to check for singularity by looking
    # at a11.  Note the choice of eps is reasonable but somewhat arbitrary.
    if abs(c11) < 256.0 * 2.220446049250313e-16:
        return False
    if c11 != 0.0:
        x[1] = d1 / c11
    else:
        return False
    if c00 != 0.0:
        x[0] = (d0 - c01 * x[1]) / c00
    else:
        return False
    # other failures in solution?
    if not (x[0] < VTK_DOUBLE_MAX and x[0] > -VTK_DOUBLE_MAX):
        return False
    if not (x[1] < VTK_DOUBLE_MAX and x[1] > -VTK_DOUBLE_MAX):
        return False
    # If necessary swap solution vector rows.
    if cols_swapped:
        var t = x[0]
        x[0] = x[1]
        x[1] = t
    return True


# ===================================================================
# Common/DataModel/vtkCellType.h
# ===================================================================

comptime VTK_EMPTY_CELL = 0
comptime VTK_VERTEX = 1
comptime VTK_POLY_VERTEX = 2
comptime VTK_LINE = 3
comptime VTK_POLY_LINE = 4
comptime VTK_TRIANGLE = 5
comptime VTK_TRIANGLE_STRIP = 6
comptime VTK_POLYGON = 7
comptime VTK_PIXEL = 8
comptime VTK_QUAD = 9
comptime VTK_TETRA = 10
comptime VTK_VOXEL = 11
comptime VTK_HEXAHEDRON = 12
comptime VTK_WEDGE = 13
comptime VTK_PYRAMID = 14
comptime VTK_PENTAGONAL_PRISM = 15
comptime VTK_HEXAGONAL_PRISM = 16
comptime VTK_INVALID = 1000

# vtkCellTypeUtilities::GetDimension
def GetDimension(cellType: Int) -> Int:
    if cellType == VTK_EMPTY_CELL:
        return -1
    if cellType == VTK_VERTEX or cellType == VTK_POLY_VERTEX:
        return 0
    if cellType == VTK_LINE or cellType == VTK_POLY_LINE:
        return 1
    if (
        cellType == VTK_TRIANGLE
        or cellType == VTK_TRIANGLE_STRIP
        or cellType == VTK_POLYGON
        or cellType == VTK_PIXEL
        or cellType == VTK_QUAD
    ):
        return 2
    if (
        cellType == VTK_TETRA
        or cellType == VTK_VOXEL
        or cellType == VTK_HEXAHEDRON
        or cellType == VTK_WEDGE
        or cellType == VTK_PYRAMID
        or cellType == VTK_PENTAGONAL_PRISM
        or cellType == VTK_HEXAGONAL_PRISM
    ):
        return 3
    return -1


# vtkPolyData reports a poly of three points as VTK_TRIANGLE and one of four
# as VTK_QUAD; everything else is VTK_POLYGON.
def PolyCellType(npts: Int) -> Int:
    if npts == 3:
        return VTK_TRIANGLE
    if npts == 4:
        return VTK_QUAD
    return VTK_POLYGON


# ===================================================================
# Common/DataModel/vtkCellArray.h -- the (connectivity, offsets) pair
# ===================================================================


def GetNumberOfIds(offsets: PtrI, cell: Int) -> Int:
    return Int(offsets[cell + 1] - offsets[cell])


def GetId(conn: PtrI, offsets: PtrI, cell: Int, i: Int) -> Int:
    return Int(conn[Int(offsets[cell]) + i])


# ===================================================================
# Common/DataModel/vtkLine.cxx
# ===================================================================

comptime VTK_NO_INTERSECT = 0
comptime VTK_INTERSECT = 1
comptime VTK_ON_LINE = 2
comptime Absolute = 0
comptime Relative = 1
comptime AbsoluteFuzzy = 2
comptime RelativeFuzzy = 3


# `t` is a one-element scratch slot, `w` a 6-double scratch window; the return
# value is the distance.
def DistanceToLine(x: PtrF, p1: PtrF, p2: PtrF, t: PtrF, w: PtrF) -> Float64:
    for i in range(3):
        w[i] = p2[i] - p1[i]

    #   Get parametric location
    var num = w[0] * (x[0] - p1[0]) + w[1] * (x[1] - p1[1]) + w[2] * (x[2] - p1[2])
    var endpoint = -1  # 0 -> p1, 1 -> p2
    var tt = 0.0
    if num == 0.0:
        t[0] = 0.0
        endpoint = 0
    else:
        var denom = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
        # trying to avoid an expensive std::abs
        var tolerance = VTK_TOL * num
        if tolerance < 0.0:
            tolerance = -tolerance
        if denom < tolerance:  # numerically bad!
            if num > 0.0:
                endpoint = 1
                t[0] = VTK_DOUBLE_MAX
            else:
                endpoint = 0
                t[0] = VTK_DOUBLE_MIN
        #
        # If parametric coordinate is within 0<=p<=1, then the point is closest
        # to the line.  Otherwise, it's closest to a point at the end.
        #
        else:
            tt = num / denom
            t[0] = tt
            if tt < 0.0:
                endpoint = 0
            elif tt > 1.0:
                endpoint = 1
            else:
                var c = w + 3
                for i in range(3):
                    c[i] = p1[i] + tt * w[i]
                return Distance2BetweenPoints(x, c) ** 0.5
    if endpoint == 0:
        return Distance2BetweenPoints(x, p1) ** 0.5
    return Distance2BetweenPoints(x, p2) ** 0.5


def Intersection(
    a1: PtrF,
    a2: PtrF,
    b1: PtrF,
    b2: PtrF,
    uv: PtrF,
    tolerance: Float64,
    tolType: Int,
    w: PtrF,
) -> Int:
    var a21 = w
    var b21 = w + 3
    var b1a1 = w + 6
    var ptu = w + 9
    var ptv = w + 12
    var tt = w + 15
    var p = w + 16
    var l1 = w + 19
    var l2 = w + 22

    for i in range(3):
        a21[i] = a2[i] - a1[i]
        b21[i] = b2[i] - b1[i]
        b1a1[i] = b1[i] - a1[i]

    #  Initialize
    uv[0] = 0.0
    uv[1] = 0.0

    #   Compute the system (least squares) matrix.
    var row1_0 = SquaredNorm(a21)
    var row1_1 = -Dot(a21, b21)
    var row2_1 = SquaredNorm(b21)

    #   Compute the least squares system constant term.
    var c0 = Dot(a21, b1a1)
    var c1 = -Dot(b21, b1a1)

    #  Solve the system of equations. Check for collinearity.
    if not SolveLinearSystemGEPP2x2(row1_0, row1_1, row1_1, row2_1, c0, c1, uv):
        # The lines are collinear. Therefore, one of the four endpoints is the
        # point of closest approach
        var minDist = VTK_DOUBLE_MAX
        for i in range(4):
            var q = a1
            var r1 = b1
            var r2 = b2
            if (i & 1) != 0:
                q = a2
            if i >= 2:
                r1 = a1
                r2 = a2
            for k in range(3):
                p[k] = q[k]
            var dist = DistanceToLine(p, r1, r2, tt, w + 25)
            if dist < minDist:
                minDist = dist
                if i < 2:
                    uv[1] = tt[0]
                    uv[0] = Float64(i % 2)
                else:
                    uv[0] = tt[0]
                    uv[1] = Float64(i % 2)
        return VTK_ON_LINE

    # The lines are not collinear, check for intersection.  However if they are
    # nearly parallel then the solution found by vtkMath::SolveLinearSystem may
    # be very inaccurate.  We hence need to check the solution against a
    # tolerance criterion.
    var u = uv[0]
    var v = uv[1]
    for i in range(3):
        ptu[i] = a21[i] * u + a1[i]
        ptv[i] = b21[i] * v + b1[i]
    var diff2 = 0.0
    for i in range(3):
        var dd = ptu[i] - ptv[i]
        diff2 += dd * dd

    var tol2 = 0.0
    if tolerance < VTK_DOUBLE_MAX:
        # compare either absolute or relative diff; hence either
        # tolerance*tolerance or diff > tolerance * max(nrm(ptu),nrm(ptv))
        # but without taking square roots.
        if tolType == Absolute or tolType == AbsoluteFuzzy:
            tol2 = tolerance * tolerance
        else:
            var n1 = SquaredNorm(ptv)
            var n2 = SquaredNorm(ptu)
            tol2 = tolerance * tolerance * (n1 if n1 > n2 else n2)
        if diff2 > tol2:
            return VTK_NO_INTERSECT

    # Check parametric coordinates for intersection within the two finite line
    # segments specified.  Special treatment is required to make sure that
    # intersections near the line end points are not within tolerance.
    if (0.0 <= u) and (u <= 1.0) and (0.0 <= v) and (v <= 1.0):
        return VTK_INTERSECT
    # Otherwise the intersection may be within tolerance at one or both of the
    # line end points.
    elif tolType >= RelativeFuzzy and tol2 > 0.0:
        var uTol = sqrt(tol2 / SquaredNorm(a21))
        var vTol = sqrt(tol2 / SquaredNorm(b21))
        if (-uTol <= u) and (u <= (1.0 + uTol)) and (-vTol <= v) and (v <= (1.0 + vTol)):
            return VTK_INTERSECT

    return VTK_NO_INTERSECT


# ===================================================================
# Common/DataModel/vtkPolygon.cxx
# ===================================================================


# Anonymous-namespace `NormalWorker`: for every segment P_i P_j (j = i+1)
# anchored at a point C, accumulate (P_i - C) x (P_j - C).
def NormalWorker(pts: PtrF, numPts: Int, ids: PtrI, useIds: Int, n: PtrF, w: PtrF):
    var v1 = w
    var v2 = w + 3
    var a = w + 6
    var p0 = w + 9
    var pointId = 0
    var commonPointId = -1
    while pointId < (numPts - 2):
        var i0 = pointId
        var i1 = pointId + 1
        if useIds != 0:
            i0 = Int(ids[pointId])
            i1 = Int(ids[pointId + 1])
        for k in range(3):
            p0[k] = pts[3 * i0 + k]
            a[k] = pts[3 * i1 + k]
        Subtract(a, p0, v1)
        if SquaredNorm(v1) > 0.0:
            commonPointId = pointId
            # consume the two points we just used to obtain a non-zero v1
            pointId += 2
            break
        pointId += 1

    if pointId >= numPts or commonPointId < 0:
        # Either all the points in the loop were coincident or we used
        # all the points to obtain v1 and have nothing left for v2.
        n[0] = 0.0
        n[1] = 0.0
        n[2] = 0.0
        return

    var c0 = commonPointId
    if useIds != 0:
        c0 = Int(ids[commonPointId])
    for k in range(3):
        p0[k] = pts[3 * c0 + k]
    while pointId < numPts:
        var i0 = pointId
        if useIds != 0:
            i0 = Int(ids[pointId])
        for k in range(3):
            a[k] = pts[3 * i0 + k]
        pointId += 1
        Subtract(a, p0, v2)
        Cross(v1, v2, v1)
        Add(n, v1, n)
        for k in range(3):
            var t = v1[k]
            v1[k] = v2[k]
            v2[k] = t


# Common/DataModel/vtkCellStatus.h -- bit flags, not an enumeration
comptime VTK_CELL_VALID: Int = 0x00
comptime VTK_CELL_WRONG_NUMBER_OF_POINTS: Int = 0x01
comptime VTK_CELL_INTERSECTING_EDGES: Int = 0x02
comptime VTK_CELL_INTERSECTING_FACES: Int = 0x04
comptime VTK_CELL_NONCONTIGUOUS_EDGES: Int = 0x08
comptime VTK_CELL_NONCONVEX: Int = 0x10
comptime VTK_CELL_FACES_ARE_ORIENTED_INCORRECTLY: Int = 0x20
comptime VTK_CELL_NONPLANAR_FACES: Int = 0x40
comptime VTK_CELL_DEGENERATE_FACES: Int = 0x80
comptime VTK_CELL_COINCIDENT_POINTS: Int = 0x100


def ComputeNormal(pts: PtrF, numPts: Int, ids: PtrI, useIds: Int, n: PtrF, w: PtrF) -> Int:
    n[0] = 0.0
    n[1] = 0.0
    n[2] = 0.0
    if numPts < 3:
        return VTK_CELL_WRONG_NUMBER_OF_POINTS

    NormalWorker(pts, numPts, ids, useIds, n, w)

    if Normalize(n) != 0.0:
        return VTK_CELL_VALID
    return VTK_CELL_DEGENERATE_FACES


# `vtkPolygon::ComputeTolerance` with the bounds walk it depends on.
def ComputeTolerance(pts: PtrF, numPts: Int, tolerance: Float64, w: PtrF) -> Float64:
    var lo = w
    var hi = w + 3
    for k in range(3):
        lo[k] = VTK_DOUBLE_MAX
        hi[k] = -VTK_DOUBLE_MAX
    for i in range(numPts):
        for k in range(3):
            var v = pts[3 * i + k]
            if v < lo[k]:
                lo[k] = v
            if v > hi[k]:
                hi[k] = v
    var d = sqrt(
        (hi[0] - lo[0]) * (hi[0] - lo[0]) + (hi[1] - lo[1]) * (hi[1] - lo[1]) + (
            hi[2] - lo[2]
        ) * (hi[2] - lo[2])
    )
    return tolerance * d


# `SimpleTriangulation`: 3 and 4 points, including the concave quad.
def SimpleTriangulation(pts: PtrF, numPts: Int, tol2: Float64, tris: PtrI, w: PtrF) -> Bool:
    var number_of_verts = numPts
    # x holds the four polygon points consecutively
    var x = w
    var d1 = w + 12
    var d2 = w + 15
    var v1 = w + 18
    var v3 = w + 21
    var normal = w + 24
    var n1 = w + 27
    var n2 = w + 30

    # the four polygon points, gathered the way vtkPolygon does it
    for i in range(4):
        if i < numPts:
            for k in range(3):
                x[3 * i + k] = pts[3 * i + k]

    # Just output the single triangle
    if number_of_verts == 3:
        var valid = True
        if (
            Distance2BetweenPoints(x, x + 3) < tol2
            or Distance2BetweenPoints(x + 3, x + 6) < tol2
            or Distance2BetweenPoints(x, x + 6) < tol2
        ):
            valid = False
        if valid:
            tris[0] = 0
            tris[1] = 1
            tris[2] = 2
            return True
    # Four points are split into two triangles. Watch out for the
    # concave case (i.e., quad looks like a arrowhead).
    elif number_of_verts == 4:
        # There are only two ear cutting possibility.  This boolean
        var use_d1 = True
        var concave = False
        var area = 0.0

        # Build diagonals for ear cutting
        for k in range(3):
            d1[k] = x[6 + k] - x[k]
            d2[k] = x[9 + k] - x[3 + k]

        var d1_n2 = SquaredNorm(d1)
        var d2_n2 = SquaredNorm(d2)
        if d1_n2 < d2_n2:
            use_d1 = True
            # check diagonal validity
            if d1_n2 < tol2:
                return False
            # prepare vector for fan building
            for k in range(3):
                v1[k] = x[3 + k] - x[k]
                v3[k] = x[9 + k] - x[k]
        else:
            use_d1 = False
            # check diagonal validity
            if d2_n2 < tol2:
                return False
            # prepare vector for fan building
            for k in range(3):
                v1[k] = x[6 + k] - x[3 + k]
                v3[k] = x[0 + k] - x[3 + k]
        # Check points validity
        if SquaredNorm(v1) < tol2:
            return False
        if SquaredNorm(v3) < tol2:
            return False
        # build polygon normal to get coherent result with earcut algo
        if use_d1:
            Cross(v1, d1, n1)
            Cross(d1, v3, n2)
        else:
            Cross(v1, d2, n1)
            Cross(d2, v3, n2)
        # Indirect check points validity
        if SquaredNorm(n1) < tol2:
            return False
        if SquaredNorm(n2) < tol2:
            return False
        # Now finalize the normal building
        for k in range(3):
            normal[k] = n1[k] + n2[k]
        if Normalize(normal) == 0.0:
            return False

        # check for concave or invalid case
        area = Dot(n1, normal)
        if area < 0.0:
            concave = True
        elif area == 0.0:
            return False
        else:
            area = Dot(n2, normal)
            if area < 0.0:
                concave = True
            elif area == 0.0:
                return False
        # Best possible case has concavity.  Try the opposite case.
        if concave:
            use_d1 = not concave  # switch use_d1 if concave is true
            # - arrowhead is OK
            # - self intersecting like quad is KO
            if use_d1:
                for k in range(3):
                    v1[k] = x[3 + k] - x[k]
                    v3[k] = x[9 + k] - x[k]
                Cross(v1, d2, n1)
                Cross(d2, v3, n2)
            else:
                for k in range(3):
                    v1[k] = x[6 + k] - x[3 + k]
                    v3[k] = x[0 + k] - x[3 + k]
                Cross(v1, d1, n1)
                Cross(d1, v3, n2)
            # Check points validity
            if SquaredNorm(v1) < tol2:
                return False
            if SquaredNorm(v3) < tol2:
                return False
            # check for invalid case
            if Dot(n1, normal) <= 0.0:
                return False
            if Dot(n2, normal) <= 0.0:
                return False

        # Finalize the tris
        # constexpr std::array<vtkIdType, 6> localPtIds{...}
        if use_d1:
            tris[0] = 0
            tris[1] = 1
            tris[2] = 2
            tris[3] = 0
            tris[4] = 2
            tris[5] = 3
        else:
            tris[0] = 0
            tris[1] = 1
            tris[2] = 3
            tris[3] = 1
            tris[4] = 2
            tris[5] = 3
        return True
    return False


# ===================================================================
# Common/Core/vtkPriorityQueue.cxx
#
# Flat binary heap.  `priority`/`qid` hold the tree, `itemLoc` maps an id to its
# slot (-1 when absent), `queueState` is [maxId, itemLoc high water, capacity].
# Pop order is identical to vtkPriorityQueue's, which is what makes the ear-cut
# output identical.
# ===================================================================


def PriorityQueueInsert(
    priority: PtrF, qid: PtrI, itemLoc: PtrI, queueState: PtrI, prio: Float64, id: Int
):
    var maxId = Int(queueState[0])
    var high = Int(queueState[1])
    # check and make sure item hasn't been inserted before
    if id <= high and Int(itemLoc[id]) != -1:
        return
    # start by placing new entry at bottom of tree
    maxId += 1
    if maxId >= Int(queueState[2]):
        return
    queueState[0] = Int64(maxId)
    priority[maxId] = prio
    qid[maxId] = Int64(id)

    var oldHigh = high
    itemLoc[id] = Int64(maxId)
    if id > high:
        queueState[1] = Int64(id)
    for i in range(oldHigh + 1, id):
        # initialize previously unused elements
        itemLoc[i] = -1

    # now begin percolating towards top of tree
    var i2 = maxId
    while i2 > 0:
        var idx = (i2 - 1) // 2
        if not (priority[i2] < priority[idx]):
            break
        var tp = priority[i2]
        priority[i2] = priority[idx]
        priority[idx] = tp
        var tq = qid[i2]
        itemLoc[tq] = Int64(idx)
        qid[i2] = qid[idx]
        itemLoc[qid[idx]] = Int64(i2)
        qid[idx] = tq
        i2 = idx


def PriorityQueuePop(
    priority: PtrF, qid: PtrI, itemLoc: PtrI, queueState: PtrI, location: Int
) -> Int:
    var maxId = Int(queueState[0])
    if maxId < 0:
        return -1

    var id = Int(qid[location])

    # move the last item to the location specified and push into the tree
    qid[location] = qid[maxId]
    priority[location] = priority[maxId]
    itemLoc[qid[location]] = Int64(location)
    itemLoc[id] = -1

    maxId -= 1
    queueState[0] = Int64(maxId)
    if maxId <= 0:
        return id

    # percolate down the tree from the specified location
    var lastNodeToCheck = (maxId - 1) // 2
    var i3 = location
    while i3 <= lastNodeToCheck:
        var idx = 2 * i3 + 1
        var j = idx
        if idx != maxId and not (priority[idx] < priority[idx + 1]):
            j = idx + 1
        if priority[i3] > priority[j]:
            var tp = priority[i3]
            priority[i3] = priority[j]
            priority[j] = tp
            var tq = qid[i3]
            itemLoc[tq] = Int64(j)
            qid[i3] = qid[j]
            itemLoc[qid[j]] = Int64(i3)
            qid[j] = tq
        else:
            break
        i3 = j

    # percolate up the tree from the specified location
    var i4 = location
    while i4 > 0:
        var idx = (i4 - 1) // 2
        if priority[i4] < priority[idx]:
            var tp = priority[i4]
            priority[i4] = priority[idx]
            priority[idx] = tp
            var tq = qid[i4]
            itemLoc[tq] = Int64(idx)
            qid[i4] = qid[idx]
            itemLoc[qid[idx]] = Int64(i4)
            qid[idx] = tq
        else:
            break
        i4 = idx

    return id


def PriorityQueueDeleteId(
    priority: PtrF, qid: PtrI, itemLoc: PtrI, queueState: PtrI, id: Int
) -> Float64:
    var high = Int(queueState[1])
    if id <= high:
        var loc = Int(itemLoc[id])
        if loc != -1:
            var prio = priority[loc]
            PriorityQueuePop(priority, qid, itemLoc, queueState, loc)
            return prio
    return VTK_DOUBLE_MAX


def PriorityQueueGetNumberOfItems(queueState: PtrI) -> Int:
    return Int(queueState[0]) + 1


# ===================================================================
# Common/DataModel/vtkPolygon.cxx -- vtkPolyVertexList
#
# `verts` is a flat array of numVerts records of VTX_STRIDE doubles laid out as
# [id, x0, x1, x2, measure, next, previous].  `state` is [head, numberOfVerts]
# and `normal` is the list normal.
# ===================================================================

comptime VTX_STRIDE = 7
comptime VTX_ID = 0
comptime VTX_X = 1
comptime VTX_MEASURE = 4
comptime VTX_NEXT = 5
comptime VTX_PREV = 6


def PolyVertexListNew(
    pts: PtrF, numVerts: Int, tol2: Float64, verts: PtrF, state: PtrI
) -> Int:
    for i in range(numVerts):
        var b = i * VTX_STRIDE
        verts[b + VTX_ID] = Float64(i)
        for k in range(3):
            verts[b + VTX_X + k] = pts[3 * i + k]
        verts[b + VTX_NEXT] = Float64(i + 1)
        verts[b + VTX_PREV] = Float64(i - 1)
        if i == (numVerts - 1):
            verts[b + VTX_NEXT] = Float64(0)
        if i == 0:
            verts[b + VTX_PREV] = Float64(numVerts - 1)

    # Make sure that there are no coincident vertices.
    # Beware of multiple coincident vertices.
    var head = 0
    var numberOfVerts = numVerts
    var vtx = 0
    for _ in range(numVerts):
        var vtxb = vtx * VTX_STRIDE
        var nxt = Int(verts[vtxb + VTX_NEXT])
        var nxtb = nxt * VTX_STRIDE
        var nxtnxt = Int(verts[nxtb + VTX_NEXT])
        if Distance2BetweenPoints(verts + vtxb + VTX_X, verts + nxtb + VTX_X) < tol2:
            verts[nxtnxt * VTX_STRIDE + VTX_PREV] = Float64(vtx)
            verts[vtxb + VTX_NEXT] = Float64(nxtnxt)
            if nxt == head:
                head = vtx
            numberOfVerts -= 1
        else:  # can move forward
            vtx = nxt
    state[0] = Int64(head)
    state[1] = Int64(numberOfVerts)
    return numberOfVerts


def PolyVertexListComputeNormal(verts: PtrF, state: PtrI, normal: PtrF, w: PtrF) -> Bool:
    var v1 = w
    var v2 = w + 3
    var n = w + 6
    var head = Int(state[0])
    normal[0] = 0.0
    normal[1] = 0.0
    normal[2] = 0.0
    var hb = head * VTX_STRIDE
    var vtx = Int(verts[hb + VTX_NEXT])
    var guard = 0
    while Int(verts[vtx * VTX_STRIDE + VTX_NEXT]) != head and guard < 1 << 22:
        guard += 1
        var vb = vtx * VTX_STRIDE
        var nb = Int(verts[vb + VTX_NEXT]) * VTX_STRIDE
        for k in range(3):
            v1[k] = verts[vb + VTX_X + k] - verts[hb + VTX_X + k]
            v2[k] = verts[nb + VTX_X + k] - verts[hb + VTX_X + k]
        Cross(v1, v2, n)
        for k in range(3):
            normal[k] += n[k]
        vtx = Int(verts[vb + VTX_NEXT])
    return Normalize(normal) != 0.0


# PERIMETER2_TO_AREA_RATIO, the only measure a filter in this port can reach.
def PolyVertexListComputeMeasure(verts: PtrF, normal: PtrF, i: Int, w: PtrF) -> Float64:
    var vb = i * VTX_STRIDE
    var pb = Int(verts[vb + VTX_PREV]) * VTX_STRIDE
    var nb = Int(verts[vb + VTX_NEXT]) * VTX_STRIDE
    var v1 = w
    var v2 = w + 3
    var v3 = w + 6
    var v4 = w + 9
    for k in range(3):
        v1[k] = verts[vb + VTX_X + k] - verts[pb + VTX_X + k]
        v2[k] = verts[nb + VTX_X + k] - verts[vb + VTX_X + k]
        v3[k] = verts[pb + VTX_X + k] - verts[nb + VTX_X + k]
    Cross(v1, v2, v4)  # |v4| is twice the area
    var area = Dot(v4, normal)
    if area < 0.0:
        verts[vb + VTX_MEASURE] = -1.0  # concave or bad triangle
        return -1.0
    elif area == 0.0:
        verts[vb + VTX_MEASURE] = -VTK_DOUBLE_MAX  # concave or bad triangle
        return -VTK_DOUBLE_MAX

    # If here, the vertex is convex and the area of the triangle is positive.
    var perimeter = Norm(v1) + Norm(v2) + Norm(v3)
    var m = perimeter * perimeter / area
    verts[vb + VTX_MEASURE] = m
    return m


def PolyVertexListCanRemoveVertex(
    verts: PtrF, state: PtrI, normal: PtrF, tol: Float64, currentVtx: Int, w: PtrF
) -> Bool:
    # Check for simple case
    if Int(state[1]) <= 3:
        return True

    # Compute split plane, the point to be cut off
    # is always on the positive side of the plane.
    var v = w
    var sN = w + 3
    var uv = w + 6
    var cb = currentVtx * VTX_STRIDE
    var previous = Int(verts[cb + VTX_PREV])
    var pb = previous * VTX_STRIDE
    var nxt = Int(verts[cb + VTX_NEXT])
    var nb = nxt * VTX_STRIDE

    var sPt = verts + pb + VTX_X  # point on plane
    for k in range(3):
        v[k] = verts[nb + VTX_X + k] - verts[pb + VTX_X + k]

    Cross(v, normal, sN)
    if Normalize(sN) == 0.0:
        return False  # bad split, indeterminant

    # Traverse the other points to see if a) they are all on the
    # other side of the plane; and if not b) whether they intersect
    # the split line.
    var oneNegative = 0
    var nextnext = Int(verts[nb + VTX_NEXT])
    var val = Dot(sN, verts + nextnext * VTX_STRIDE + VTX_X) - Dot(sN, sPt)
    var currentSign = 0
    if val > tol:
        currentSign = 1
    elif val < -tol:
        currentSign = -1
    oneNegative = 0
    if currentSign < 0:
        oneNegative = 1  # very important

    # Intersections are only computed when the split half-space is crossed
    var vtx = Int(verts[nextnext * VTX_STRIDE + VTX_NEXT])
    var guard = 0
    while vtx != previous and guard < 1 << 22:
        guard += 1
        var vb = vtx * VTX_STRIDE
        val = Dot(sN, verts + vb + VTX_X) - Dot(sN, sPt)
        var sign = 0
        if val > tol:
            sign = 1
        elif val < -tol:
            sign = -1
        if sign != currentSign:
            if not oneNegative:
                if sign < 0:
                    oneNegative = 1  # very important
            if (
                Intersection(
                    sPt,
                    verts + nb + VTX_X,
                    verts + vb + VTX_X,
                    verts + Int(verts[vb + VTX_PREV]) * VTX_STRIDE + VTX_X,
                    uv,
                    tol,
                    AbsoluteFuzzy,
                    w + 8,
                ) != 0
            ):
                return False
            else:
                currentSign = sign
        vtx = Int(verts[vb + VTX_NEXT])
    if not oneNegative:
        return False  # entire loop is on this side of plane
    return True


def PolyVertexListRemoveVertex(
    verts: PtrF,
    state: PtrI,
    normal: PtrF,
    queuePriority: PtrF,
    queueId: PtrI,
    itemLoc: PtrI,
    queueState: PtrI,
    i: Int,
    tris: PtrI,
    trisCount: PtrI,
    w: PtrF,
):
    # Create triangle
    var vb = i * VTX_STRIDE
    var nb = Int(verts[vb + VTX_NEXT])
    var pb = Int(verts[vb + VTX_PREV])
    tris[trisCount[0]] = Int64(verts[vb + VTX_ID])
    tris[trisCount[0] + 1] = Int64(verts[nb * VTX_STRIDE + VTX_ID])
    tris[trisCount[0] + 2] = Int64(verts[pb * VTX_STRIDE + VTX_ID])
    trisCount[0] += 3

    # remove vertex; special case if single triangle left
    state[1] = Int64(Int(state[1]) - 1)
    if Int(state[1]) < 3:
        return
    if Int(state[0]) == i:
        state[0] = Int64(nb)
    verts[pb * VTX_STRIDE + VTX_NEXT] = Float64(nb)
    verts[nb * VTX_STRIDE + VTX_PREV] = Float64(pb)

    # recompute measure, reinsert into queue
    # note that id may have been previously deleted (with Pop()) if we
    # are dealing with a concave polygon and vertex couldn't be split.
    PriorityQueueDeleteId(queuePriority, queueId, itemLoc, queueState, pb)
    PriorityQueueDeleteId(queuePriority, queueId, itemLoc, queueState, nb)
    if PolyVertexListComputeMeasure(verts, normal, pb, w) > 0.0:
        PriorityQueueInsert(
            queuePriority,
            queueId,
            itemLoc,
            queueState,
            verts[pb * VTX_STRIDE + VTX_MEASURE],
            pb,
        )
    if PolyVertexListComputeMeasure(verts, normal, nb, w) > 0.0:
        PriorityQueueInsert(
            queuePriority,
            queueId,
            itemLoc,
            queueState,
            verts[nb * VTX_STRIDE + VTX_MEASURE],
            nb,
        )


# `vtkPolygon::EarCutTriangulation` (this port only ever asks for
# PERIMETER2_TO_AREA_RATIO, the default, so the other two measures and the
# BoundedTriangulate retry path are not reachable and are not ported).
def EarCutTriangulation(
    pts: PtrF,
    numPts: Int,
    tol2: Float64,
    verts: PtrF,
    state: PtrI,
    normal: PtrF,
    queuePriority: PtrF,
    queueId: PtrI,
    itemLoc: PtrI,
    queueState: PtrI,
    tris: PtrI,
    trisCount: PtrI,
    w: PtrF,
) -> Bool:
    trisCount[0] = 0

    # Make sure there are at least 3 vertices
    if numPts < 3:
        return False

    # Check for trivial triangulation cases
    if SimpleTriangulation(pts, numPts, tol2, tris, w):
        if numPts == 3:
            trisCount[0] = 3
        else:
            trisCount[0] = 6
        return True

    # Establish a more convenient structure for the triangulation process
    PolyVertexListNew(pts, numPts, tol2, verts, state)

    # The polygon normal is needed during triangulation
    if not PolyVertexListComputeNormal(verts, state, normal, w + 33):
        return False

    # Now compute the angles between edges incident to each vertex.  Place the
    # structure into a priority queue (those vertices with smallest measure are
    # to be removed first).
    # vtkPriorityQueue::Reset() clears the tree; ItemLocation is filled with
    # -1 by Allocate(), which is what makes Insert's "already queued?" test work.
    queueState[0] = -1
    queueState[1] = -1
    for i in range(numPts):
        itemLoc[i] = -1
    var vtx = Int(state[0])
    for _ in range(Int(state[1])):
        # concave (negative measure) vertices are not eligible for removal
        if PolyVertexListComputeMeasure(verts, normal, vtx, w) > 0.0:
            PriorityQueueInsert(
                queuePriority,
                queueId,
                itemLoc,
                queueState,
                verts[vtx * VTX_STRIDE + VTX_MEASURE],
                vtx,
            )
        vtx = Int(verts[vtx * VTX_STRIDE + VTX_NEXT])

    # For each vertex in the priority queue, and as long as there are three or
    # more vertices, remove the vertex (if possible) and create a new triangle.
    var tol = sqrt(tol2)
    while Int(state[1]) > 2 and PriorityQueueGetNumberOfItems(queueState) > 0:
        var id = PriorityQueuePop(queuePriority, queueId, itemLoc, queueState, 0)
        if id < 0:
            break
        # removes it, even if can't be split
        if PolyVertexListCanRemoveVertex(verts, state, normal, tol, id, w + 9):
            PolyVertexListRemoveVertex(
                verts,
                state,
                normal,
                queuePriority,
                queueId,
                itemLoc,
                queueState,
                id,
                tris,
                trisCount,
                w,
            )
    if Int(state[1]) > 2:  # couldn't triangulate
        return False
    return True


# `vtkQuad::TriangulateLocalIds`: the shorter diagonal wins.
def QuadTriangulateLocalIds(pts: PtrF, tris: PtrI):
    var d1 = Distance2BetweenPoints(pts, pts + 6)
    var d2 = Distance2BetweenPoints(pts + 3, pts + 9)
    if d1 <= d2:
        tris[0] = 0
        tris[1] = 1
        tris[2] = 2
        tris[3] = 0
        tris[4] = 2
        tris[5] = 3
    else:
        tris[0] = 0
        tris[1] = 1
        tris[2] = 3
        tris[3] = 1
        tris[4] = 2
        tris[5] = 3


# `vtkPolygon::TriangulateLocalIds`.  `polygonPts` holds the numPts points of
# the polygon gathered out of the dataset, `tris` receives local ids.
def TriangulateLocalIds(
    polygonPts: PtrF,
    numPts: Int,
    tolerance: Float64,
    verts: PtrF,
    state: PtrI,
    normal: PtrF,
    queuePriority: PtrF,
    queueId: PtrI,
    itemLoc: PtrI,
    queueState: PtrI,
    tris: PtrI,
    trisCount: PtrI,
    w: PtrF,
) -> Bool:
    var successful = True
    var tol = ComputeTolerance(polygonPts, numPts, tolerance, w)
    var success = EarCutTriangulation(
        polygonPts,
        numPts,
        tol * tol,
        verts,
        state,
        normal,
        queuePriority,
        queueId,
        itemLoc,
        queueState,
        tris,
        trisCount,
        w,
    )
    if not success:  # Indicate possible failure
        # Non-planar quads can cause the ear cut triangulation algorithm to
        # fail.  If that happens, use vtkQuad specifically to triangulate it.
        if numPts == 4:
            QuadTriangulateLocalIds(polygonPts, tris)
            trisCount[0] = 6
        else:
            # vtkDebugMacro(<< "Possible triangulation failure");
            pass
        successful = False
    return successful


# ===================================================================
# Common/DataModel/vtkTriangleStrip.cxx
# ===================================================================


# Given a triangle strip, decompose it into (triangle) polygons.  The polygons
# are appended to the end of the list of polygons, starting at cell `cellId`;
# `connOffset` is the connectivity offset of cell `cellId` and is advanced.
def DecomposeStrip(
    npts: Int, pts: PtrI, oconn: PtrI, ooffsets: PtrI, cellId: Int, connOffset: PtrI
) -> Int:
    var cid = cellId
    var p1 = pts[0]
    var p2 = pts[1]
    for i in range(npts - 2):
        var p3 = pts[i + 2]
        if (i % 2) != 0:  # flip ordering to preserve consistency
            oconn[connOffset[0]] = p2
            oconn[connOffset[0] + 1] = p1
            oconn[connOffset[0] + 2] = p3
        else:
            oconn[connOffset[0]] = p1
            oconn[connOffset[0] + 1] = p2
            oconn[connOffset[0] + 2] = p3
        connOffset[0] += 3
        ooffsets[cid + 1] = connOffset[0]
        cid += 1
        p1 = p2
        p2 = p3
    return cid


# ===================================================================
# Common/DataModel/Cell/*.cxx -- GetParametricCenter + EvaluateLocation
# ===================================================================


def VertexGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.0
    pcoords[1] = 0.0
    pcoords[2] = 0.0
    return 0


def VertexEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    for k in range(3):
        x[k] = pts[k]
    weights[0] = 1.0


def LineGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.0
    pcoords[2] = 0.0
    return 0


def LineEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    for i in range(3):
        x[i] = pts[i] + pcoords[0] * (pts[3 + i] - pts[i])
    weights[0] = 1.0 - pcoords[0]
    weights[1] = pcoords[0]


def PolyLineGetParametricCenter(pcoords: PtrF, numPts: Int) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.0
    pcoords[2] = 0.0
    return (numPts - 1) // 2


def PolyLineEvaluateLocation(
    pts: PtrF, numPts: Int, subId: Int, pcoords: PtrF, x: PtrF, weights: PtrF
):
    for i in range(3):
        x[i] = pts[3 * subId + i] + pcoords[0] * (pts[3 * (subId + 1) + i] - pts[3 * subId + i])
    for i in range(numPts):
        weights[i] = 0.0
    if subId >= 0:
        weights[subId] = 1.0 - pcoords[0]
        weights[subId + 1] = pcoords[0]


def PolyVertexGetParametricCenter(pcoords: PtrF, numPts: Int) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.5
    return (numPts // 2)


def PolyVertexEvaluateLocation(pts: PtrF, numPts: Int, subId: Int, x: PtrF, weights: PtrF):
    for k in range(3):
        x[k] = pts[3 * subId + k]
    for i in range(numPts):
        weights[i] = 0.0
    weights[subId] = 1.0


def TriangleGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 1.0 / 3.0
    pcoords[1] = 1.0 / 3.0
    pcoords[2] = 0.0
    return 0


def TriangleEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    weights[0] = 1.0 - pcoords[0] - pcoords[1]
    weights[1] = pcoords[0]
    weights[2] = pcoords[1]
    for i in range(3):
        x[i] = pts[i] * weights[0] + pts[3 + i] * weights[1] + pts[6 + i] * weights[2]


def QuadInterpolationFunctions(pcoords: PtrF, sf: PtrF):
    var rm = 1.0 - pcoords[0]
    var sm = 1.0 - pcoords[1]
    sf[0] = rm * sm
    sf[1] = pcoords[0] * sm
    sf[2] = pcoords[0] * pcoords[1]
    sf[3] = rm * pcoords[1]


def QuadGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.0
    return 0


def QuadEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    QuadInterpolationFunctions(pcoords, weights)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(4):
        for j in range(3):
            x[j] += pts[3 * i + j] * weights[i]


def PixelInterpolationFunctions(pcoords: PtrF, weights: PtrF):
    var rm = 1.0 - pcoords[0]
    var sm = 1.0 - pcoords[1]
    weights[0] = rm * sm
    weights[1] = pcoords[0] * sm
    weights[2] = rm * pcoords[1]
    weights[3] = pcoords[0] * pcoords[1]


def PixelGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.0
    return 0


def PixelEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    for i in range(3):
        x[i] = pts[i] + pcoords[0] * (pts[3 + i] - pts[i]) + pcoords[1] * (pts[6 + i] - pts[i])
    PixelInterpolationFunctions(pcoords, weights)


# vtkPolygon does not override vtkCell::GetParametricCenter
def PolygonGetParametricCenter(pcoords: PtrF, numPts: Int) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.5
    return 0


# `vtkPolygon::EvaluateLocation` = ParameterizePolygon + InterpolateFunctions.
def PolygonEvaluateLocation(
    pts: PtrF, numPts: Int, pcoords: PtrF, x: PtrF, weights: PtrF, w: PtrF, ids: PtrI
):
    var p0 = w
    var p10 = w + 3
    var p20 = w + 6
    var n = w + 9
    ParameterizePolygon(pts, numPts, p0, p10, p20, n, w + 12, ids)
    for i in range(3):
        x[i] = p0[i] + pcoords[0] * p10[i] + pcoords[1] * p20[i]
    PolygonInterpolateFunctions(pts, numPts, x, weights)


def ParameterizePolygon(
    pts: PtrF, numPts: Int, p0: PtrF, p10: PtrF, p20: PtrF, n: PtrF, w: PtrF, ids: PtrI
) -> Int:
    var sbounds = w
    var tbounds = w + 2
    var p = w + 4
    var p1 = w + 7
    var p2 = w + 10
    if numPts < 3:
        return 0

    #  This is a two pass process: first create a p' coordinate system
    #  that is then adjusted to ensure that the polygon points are all in
    #  the range 0<=s,t<=1.  The p' system is defined by the polygon normal,
    #  first vertex and the first edge.
    ComputeNormal(pts, numPts, ids, 0, n, w + 13)
    for i in range(3):
        p0[i] = pts[i]
        p10[i] = pts[3 + i] - pts[i]
    Cross(n, p10, p20)

    # Determine lengths of edges
    var l10 = SquaredNorm(p10)
    var l20 = SquaredNorm(p20)
    if l10 == 0.0 or l20 == 0.0:
        return 0

    #  Now evaluate all polygon points to determine min/max parametric
    #  coordinate values.  The first vertex has (s,t) = (0,0).
    sbounds[0] = 0.0
    sbounds[1] = 0.0
    tbounds[0] = 0.0
    tbounds[1] = 0.0

    for i in range(1, numPts):
        for j in range(3):
            p[j] = pts[3 * i + j] - p0[j]
        var s = Dot(p, p10) / l10
        var t = Dot(p, p20) / l20
        if s < sbounds[0]:
            sbounds[0] = s
        if s > sbounds[1]:
            sbounds[1] = s
        if t < tbounds[0]:
            tbounds[0] = t
        if t > tbounds[1]:
            tbounds[1] = t

    #  Re-evaluate coordinate system
    for i in range(3):
        p1[i] = p0[i] + sbounds[1] * p10[i] + tbounds[0] * p20[i]
        p2[i] = p0[i] + sbounds[0] * p10[i] + tbounds[1] * p20[i]
        p0[i] = p0[i] + sbounds[0] * p10[i] + tbounds[0] * p20[i]
        p10[i] = p1[i] - p0[i]
        p20[i] = p2[i] - p0[i]
    return 1


# `vtkPolygon::InterpolateFunctions`: 1/r**2 normalized sum.
def PolygonInterpolateFunctions(pts: PtrF, numPts: Int, x: PtrF, weights: PtrF):
    var sum = 0.0
    for i in range(numPts):
        weights[i] = Distance2BetweenPoints(x, pts + 3 * i)
        if weights[i] == 0.0:  # exact hit
            for j in range(numPts):
                weights[j] = 0.0
            weights[i] = 1.0
            return
        sum += weights[i]
    if sum != 0.0:
        for i in range(numPts):
            weights[i] = 1.0 / (weights[i] * sum)
    return


def TetraGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.25
    pcoords[1] = 0.25
    pcoords[2] = 0.25
    return 0


def TetraEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    var u4 = 1.0 - pcoords[0] - pcoords[1] - pcoords[2]
    for i in range(3):
        x[i] = (
            pts[3 + i] * pcoords[0]
            + pts[6 + i] * pcoords[1]
            + pts[9 + i] * pcoords[2]
            + pts[i] * u4
        )
    weights[0] = u4
    weights[1] = pcoords[0]
    weights[2] = pcoords[1]
    weights[3] = pcoords[2]


def VoxelInterpolationFunctions(pcoords: PtrF, weights: PtrF):
    var r = pcoords[0]
    var s = pcoords[1]
    var t = pcoords[2]
    var rm = 1.0 - r
    var sm = 1.0 - s
    var tm = 1.0 - t
    weights[0] = rm * sm * tm
    weights[1] = r * sm * tm
    weights[2] = rm * s * tm
    weights[3] = r * s * tm
    weights[4] = rm * sm * t
    weights[5] = r * sm * t
    weights[6] = rm * s * t
    weights[7] = r * s * t


def VoxelGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.5
    return 0


def VoxelEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    for i in range(3):
        x[i] = (
            pts[i]
            + pcoords[0] * (pts[3 + i] - pts[i])
            + pcoords[1] * (pts[6 + i] - pts[i])
            + pcoords[2] * (pts[12 + i] - pts[i])
        )
    VoxelInterpolationFunctions(pcoords, weights)


def HexahedronInterpolationFunctions(pcoords: PtrF, sf: PtrF):
    var rm = 1.0 - pcoords[0]
    var sm = 1.0 - pcoords[1]
    var tm = 1.0 - pcoords[2]
    var rmXsm = rm * sm
    var p0Xsm = pcoords[0] * sm
    var p0Xp1 = pcoords[0] * pcoords[1]
    var rmXp1 = rm * pcoords[1]
    sf[0] = rmXsm * tm
    sf[1] = p0Xsm * tm
    sf[2] = p0Xp1 * tm
    sf[3] = rmXp1 * tm
    sf[4] = rmXsm * pcoords[2]
    sf[5] = p0Xsm * pcoords[2]
    sf[6] = p0Xp1 * pcoords[2]
    sf[7] = rmXp1 * pcoords[2]


def HexahedronGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.5
    pcoords[1] = 0.5
    pcoords[2] = 0.5
    return 0


def HexahedronEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    HexahedronInterpolationFunctions(pcoords, weights)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(8):
        for j in range(3):
            x[j] += pts[3 * i + j] * weights[i]


def WedgeInterpolationFunctions(pcoords: PtrF, sf: PtrF):
    sf[0] = (1.0 - pcoords[0] - pcoords[1]) * (1.0 - pcoords[2])
    sf[1] = pcoords[0] * (1.0 - pcoords[2])
    sf[2] = pcoords[1] * (1.0 - pcoords[2])
    sf[3] = (1.0 - pcoords[0] - pcoords[1]) * pcoords[2]
    sf[4] = pcoords[0] * pcoords[2]
    sf[5] = pcoords[1] * pcoords[2]


def WedgeGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.333333
    pcoords[1] = 0.333333
    pcoords[2] = 0.5
    return 0


def WedgeEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    WedgeInterpolationFunctions(pcoords, weights)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(6):
        for j in range(3):
            x[j] += pts[3 * i + j] * weights[i]


def PyramidInterpolationFunctions(pcoords: PtrF, sf: PtrF):
    var rm = 1.0 - pcoords[0]
    var sm = 1.0 - pcoords[1]
    var tm = 1.0 - pcoords[2]
    sf[0] = rm * sm * tm
    sf[1] = pcoords[0] * sm * tm
    sf[2] = pcoords[0] * pcoords[1] * tm
    sf[3] = rm * pcoords[1] * tm
    sf[4] = pcoords[2]


def PyramidGetParametricCenter(pcoords: PtrF) -> Int:
    pcoords[0] = 0.4
    pcoords[1] = 0.4
    pcoords[2] = 0.2
    return 0


def PyramidEvaluateLocation(pts: PtrF, pcoords: PtrF, x: PtrF, weights: PtrF):
    PyramidInterpolationFunctions(pcoords, weights)
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(5):
        for j in range(3):
            x[j] += pts[3 * i + j] * weights[i]


def TriangleStripGetParametricCenter(pcoords: PtrF, numPts: Int) -> Int:
    pcoords[0] = 0.333333
    pcoords[1] = 0.333333
    pcoords[2] = 0.0
    return ((numPts - 2) // 2)


def TriangleStripEvaluateLocation(
    pts: PtrF, numPts: Int, subId: Int, pcoords: PtrF, x: PtrF, weights: PtrF
):
    # static constexpr int idx[2][3] = { { 0, 1, 2 }, { 1, 0, 2 } };
    var order = subId % 2
    var i0 = 0
    var i1 = 1
    var i2 = 2
    if order == 1:
        i0 = 1
        i1 = 0
    # pt1/pt2/pt3 are `pts + 3 * (subId + idx[order][k])`; the component index
    # stays outside the point stride.
    var pt1 = pts + 3 * (subId + i0)
    var pt2 = pts + 3 * (subId + i1)
    var pt3 = pts + 3 * (subId + i2)
    var u3 = 1.0 - pcoords[0] - pcoords[1]
    for i in range(numPts):
        weights[i] = 0.0
    weights[subId] = u3
    weights[subId + 1] = pcoords[0]
    weights[subId + 2] = pcoords[1]
    for i in range(3):
        x[i] = pt1[i] * weights[subId] + pt2[i] * weights[subId + 1] + pt3[i] * weights[subId + 2]


# `vtkGenericCell::EvaluateLocation` dispatch, driven by the cell type.
def EvaluateLocation(
    cellType: Int,
    numPts: Int,
    pts: PtrF,
    pcoords: PtrF,
    x: PtrF,
    weights: PtrF,
    w: PtrF,
    ids: PtrI,
) -> Int:
    var subId = 0
    if cellType == VTK_VERTEX:
        subId = VertexGetParametricCenter(pcoords)
        VertexEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_POLY_VERTEX:
        subId = PolyVertexGetParametricCenter(pcoords, numPts)
        PolyVertexEvaluateLocation(pts, numPts, subId, x, weights)
    elif cellType == VTK_LINE:
        subId = LineGetParametricCenter(pcoords)
        LineEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_POLY_LINE:
        subId = PolyLineGetParametricCenter(pcoords, numPts)
        PolyLineEvaluateLocation(pts, numPts, subId, pcoords, x, weights)
    elif cellType == VTK_TRIANGLE:
        subId = TriangleGetParametricCenter(pcoords)
        TriangleEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_QUAD:
        subId = QuadGetParametricCenter(pcoords)
        QuadEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_PIXEL:
        subId = PixelGetParametricCenter(pcoords)
        PixelEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_POLYGON:
        subId = PolygonGetParametricCenter(pcoords, numPts)
        PolygonEvaluateLocation(pts, numPts, pcoords, x, weights, w, ids)
    elif cellType == VTK_TRIANGLE_STRIP:
        subId = TriangleStripGetParametricCenter(pcoords, numPts)
        TriangleStripEvaluateLocation(pts, numPts, subId, pcoords, x, weights)
    elif cellType == VTK_TETRA:
        subId = TetraGetParametricCenter(pcoords)
        TetraEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_VOXEL:
        subId = VoxelGetParametricCenter(pcoords)
        VoxelEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_HEXAHEDRON:
        subId = HexahedronGetParametricCenter(pcoords)
        HexahedronEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_WEDGE:
        subId = WedgeGetParametricCenter(pcoords)
        WedgeEvaluateLocation(pts, pcoords, x, weights)
    elif cellType == VTK_PYRAMID:
        subId = PyramidGetParametricCenter(pcoords)
        PyramidEvaluateLocation(pts, pcoords, x, weights)
    else:
        return -1
    return subId


# ===================================================================
# Link / edge-table indices.  Upstream builds these once in vtkPolyData
# (vtkStaticCellLinks, vtkEdgeTable); here they are flat caller-owned arrays
# built at the top of each filter, and every consumer asks the same question
# vtkPolyData::GetPointCells / GetCellEdgeNeighbors answers.
# ===================================================================


# `GetPointCells` index: `linkOffsets` holds numPts+1 entries, `linkCells` the
# cells, `cursor` is the numPts-entry fill cursor.
def BuildPointLinks(
    conn: PtrI,
    offsets: PtrI,
    nCells: Int,
    numPts: Int,
    linkOffsets: PtrI,
    linkCells: PtrI,
    cursor: PtrI,
) -> Int:
    for i in range(numPts + 1):
        linkOffsets[i] = 0
    for c in range(nCells):
        for i in range(GetNumberOfIds(offsets, c)):
            linkOffsets[Int(conn[Int(offsets[c]) + i]) + 1] += 1
    for i in range(numPts):
        linkOffsets[i + 1] += linkOffsets[i]
    for i in range(numPts):
        cursor[i] = linkOffsets[i]
    for c in range(nCells):
        for i in range(GetNumberOfIds(offsets, c)):
            var p = Int(conn[Int(offsets[c]) + i])
            linkCells[Int(cursor[p])] = Int64(c)
            cursor[p] += 1
    return Int(linkOffsets[numPts])


# Edge table: open addressing on the packed key of the (min,max) point pair.
# `slotKeys` holds the key of a used slot (-1 when empty) and `slots` the head of
# that edge's chain; each chain entry is (cell, next) in increasing cell order,
# so GetCellEdgeNeighbors returns ids the way vtkPolyData's links do.
comptime EDGE_HASH_MUL = 2654435761


def EdgeKeyHash(k0: Int, k1: Int, slotCap: Int) -> Int:
    return Int((Int64(k0) * EDGE_HASH_MUL + Int64(k1)) % Int64(slotCap))


def EdgeKey(k0: Int, k1: Int, numPts: Int) -> Int64:
    return Int64(k0) * Int64(numPts) + Int64(k1)


def BuildEdgeTable(
    conn: PtrI,
    offsets: PtrI,
    nCells: Int,
    numPts: Int,
    slots: PtrI,
    slotKeys: PtrI,
    slotCap: Int,
    edgeCell: PtrI,
    edgeNext: PtrI,
) -> Int:
    for i in range(slotCap):
        slotKeys[i] = -1
    var numEdges = 0
    for c in range(nCells):
        var npts = GetNumberOfIds(offsets, c)
        for i in range(npts):
            var a = Int(conn[Int(offsets[c]) + i])
            var b = Int(conn[Int(offsets[c]) + (i + 1) % npts])
            var k0 = a
            var k1 = b
            if k1 < k0:
                k0 = b
                k1 = a
            var key = EdgeKey(k0, k1, numPts)
            var probe = EdgeKeyHash(k0, k1, slotCap)
            var found = -1
            while slotKeys[probe] != -1:
                if slotKeys[probe] == key:
                    found = probe
                    break
                probe = (probe + 1) % slotCap
            if found == -1:
                slotKeys[probe] = key
                slots[probe] = Int64(numEdges)
                edgeCell[numEdges] = Int64(c)
                edgeNext[numEdges] = -1
                numEdges += 1
            else:
                var tail = Int(slots[found])
                while edgeNext[tail] != -1:
                    tail = Int(edgeNext[tail])
                edgeCell[numEdges] = Int64(c)
                edgeNext[numEdges] = -1
                edgeNext[tail] = Int64(numEdges)
                numEdges += 1
    return numEdges


# `vtkPolyData::GetCellEdgeNeighbors(cell, p1, p2, adjIds)`.
def GetCellEdgeNeighbors(
    slots: PtrI,
    slotKeys: PtrI,
    slotCap: Int,
    edgeCell: PtrI,
    edgeNext: PtrI,
    numPts: Int,
    p1: Int,
    p2: Int,
    cell: Int,
    ids: PtrI,
) -> Int:
    var k0 = p1
    var k1 = p2
    if k1 < k0:
        k0 = p2
        k1 = p1
    var key = EdgeKey(k0, k1, numPts)
    var probe = EdgeKeyHash(k0, k1, slotCap)
    var found = -1
    while slotKeys[probe] != -1:
        if slotKeys[probe] == key:
            found = probe
            break
        probe = (probe + 1) % slotCap
    if found == -1:
        return 0
    var count = 0
    var e = Int(slots[found])
    while e != -1:
        var c = Int(edgeCell[e])
        if cell < 0 or c != cell:
            ids[count] = Int64(c)
            count += 1
        e = Int(edgeNext[e])
    return count


# ===================================================================
# Filters/Core/vtkTriangleFilter.cxx
# ===================================================================


# Do each of the verts, lines, polys, and strips separately.  `outSrc` records,
# for every output cell, the input cell it came from, so the caller can do
# outCD->CopyData exactly as upstream does.  `counts` is [verts, lines, polys].
def TriangleFilter(
    pts: PtrF,
    vertsConn: PtrI,
    vertsOff: PtrI,
    numInVerts: Int,
    linesConn: PtrI,
    linesOff: PtrI,
    numInLines: Int,
    polysConn: PtrI,
    polysOff: PtrI,
    numInPolys: Int,
    stripsConn: PtrI,
    stripsOff: PtrI,
    numInStrips: Int,
    passVerts: Bool,
    passLines: Bool,
    preservePolys: Bool,
    tolerance: Float64,
    maxCellSize: Int,
    oVertsConn: PtrI,
    oVertsOff: PtrI,
    oLinesConn: PtrI,
    oLinesOff: PtrI,
    oPolysConn: PtrI,
    oPolysOff: PtrI,
    outSrc: PtrI,
    counts: PtrI,
    scratchF: PtrF,
    scratchI: PtrI,
):
    # Scratch layout (see the module docstring): one float pool, one int pool.
    var w = scratchF
    var polygonPts = scratchF + 256
    var verts = scratchF + 256 + 3 * maxCellSize
    var normal = scratchF + 256 + 10 * maxCellSize
    var queuePriority = scratchF + 259 + 10 * maxCellSize
    var state = scratchI
    var queueState = scratchI + 2
    var queueId = scratchI + 5
    var itemLoc = scratchI + 5 + maxCellSize
    var tris = scratchI + 5 + 2 * maxCellSize
    var trisCount = scratchI + 5 + 5 * maxCellSize
    var connOffset = trisCount + 1

    var inCellId = 0
    var outVertId = 0
    var outVertConn = 0
    var outLineId = 0
    var outLineConn = 0
    var outPolyId = 0
    var outPolyConn = 0
    oVertsOff[0] = 0
    oLinesOff[0] = 0
    oPolysOff[0] = 0

    # verts
    if numInVerts > 0 and passVerts:
        if preservePolys:
            for i in range(Int(vertsOff[numInVerts])):
                oVertsConn[i] = vertsConn[i]
            oVertsOff[1] = vertsOff[numInVerts]
            outVertId = numInVerts
            outVertConn = Int(vertsOff[numInVerts])
            for i in range(numInVerts):
                outSrc[i] = Int64((i))
            inCellId += numInVerts
        else:
            for c in range(numInVerts):
                var npts = GetNumberOfIds(vertsOff, c)
                if npts > 1:
                    for i in range(npts):
                        oVertsConn[outVertConn] = vertsConn[Int(vertsOff[c]) + i]
                        oVertsOff[outVertId + 1] = Int64((outVertConn + 1))
                        outSrc[outVertId] = Int64((inCellId))
                        outVertConn += 1
                        outVertId += 1
                else:
                    oVertsConn[outVertConn] = vertsConn[Int(vertsOff[c])]
                    oVertsOff[outVertId + 1] = Int64((outVertConn + 1))
                    outSrc[outVertId] = Int64((inCellId))
                    outVertConn += 1
                    outVertId += 1
            inCellId += numInVerts

    # lines
    if numInLines > 0 and passLines:
        if preservePolys:
            for i in range(Int(linesOff[numInLines])):
                oLinesConn[i] = linesConn[i]
            oLinesOff[1] = linesOff[numInLines]
            outLineId = numInLines
            outLineConn = Int(linesOff[numInLines])
            for i in range(numInLines):
                outSrc[i] = Int64((i))
            inCellId += numInLines
        else:
            for c in range(numInLines):
                var npts = GetNumberOfIds(linesOff, c)
                if npts > 2:
                    for i in range(npts - 1):
                        oLinesConn[outLineConn] = linesConn[Int(linesOff[c]) + i]
                        oLinesConn[outLineConn + 1] = linesConn[Int(linesOff[c]) + i + 1]
                        oLinesOff[outLineId + 1] = Int64((outLineConn + 2))
                        outSrc[outLineId] = Int64((inCellId))
                        outLineConn += 2
                        outLineId += 1
                else:
                    oLinesConn[outLineConn] = linesConn[Int(linesOff[c])]
                    oLinesConn[outLineConn + 1] = linesConn[Int(linesOff[c]) + 1]
                    oLinesOff[outLineId + 1] = Int64((outLineConn + 2))
                    outSrc[outLineId] = Int64((inCellId))
                    outLineConn += 2
                    outLineId += 1
            inCellId += numInLines

    # polys
    for c in range(numInPolys):
        var npts = GetNumberOfIds(polysOff, c)
        if npts == 3:
            for j in range(3):
                oPolysConn[outPolyConn + j] = polysConn[Int(polysOff[c]) + j]
            oPolysOff[outPolyId + 1] = Int64((outPolyConn + 3))
            outSrc[outPolyId] = Int64((inCellId))
            outPolyConn += 3
            outPolyId += 1
        else:
            # initialize polygon
            for i in range(npts):
                var q = Int(polysConn[Int(polysOff[c]) + i])
                for k in range(3):
                    polygonPts[3 * i + k] = pts[3 * q + k]
            TriangulateLocalIds(
                polygonPts,
                npts,
                tolerance,
                verts,
                state,
                normal,
                queuePriority,
                queueId,
                itemLoc,
                queueState,
                tris,
                trisCount,
                w,
            )
            var numSimplices = Int(trisCount[0]) // 3
            for i in range(numSimplices):
                for j in range(3):
                    oPolysConn[outPolyConn + j] = polysConn[Int(polysOff[c]) + Int(tris[3 * i + j])]
                oPolysOff[outPolyId + 1] = Int64((outPolyConn + 3))
                outSrc[outPolyId] = Int64((inCellId))
                outPolyConn += 3
                outPolyId += 1
        inCellId += 1

    # strips
    connOffset[0] = Int64(outPolyConn)
    for c in range(numInStrips):
        var npts = GetNumberOfIds(stripsOff, c)
        outPolyId = DecomposeStrip(
            npts, stripsConn + Int(stripsOff[c]), oPolysConn, oPolysOff, outPolyId, connOffset
        )
        for i in range(npts - 2):
            outSrc[outPolyId - (npts - 2) + i] = Int64((inCellId))
        inCellId += 1

    counts[0] = Int64(outVertId)
    counts[1] = Int64(outLineId)
    counts[2] = Int64(outPolyId)


# ===================================================================
# Filters/Core/vtkCellCenters.cxx
# ===================================================================


# CellCenterFunctor + the empty-cell removal pass of vtkCellCenters::RequestData.
# `cellPts` is a 3*maxCellSize scratch window holding the gathered cell points.
def CellCenters(
    pts: PtrF,
    conn: PtrI,
    offsets: PtrI,
    types: PtrT,
    nCells: Int,
    centers: PtrF,
    cellCentered: PtrT,
    numPoints: PtrI,
    scratchF: PtrF,
    scratchI: PtrI,
):
    var pcoords = scratchF
    var x = scratchF + 3
    var weights = scratchF + 6
    var w = scratchF + 16
    var cellPts = scratchF + 256
    var ids = scratchI
    for cellId in range(nCells):
        var cellType = Int(types[cellId])
        for k in range(3):
            x[k] = 0.0
        if cellType != VTK_EMPTY_CELL:
            var npts = GetNumberOfIds(offsets, cellId)
            for i in range(npts):
                var q = Int(conn[Int(offsets[cellId]) + i])
                for k in range(3):
                    cellPts[3 * i + k] = pts[3 * q + k]
            EvaluateLocation(cellType, npts, cellPts, pcoords, x, weights, w, ids)
            for k in range(3):
                centers[3 * cellId + k] = x[k]
            cellCentered[cellId] = 1
        else:
            cellCentered[cellId] = 0
    # Remove points that would have been produced by empty cells
    var n = 0
    for cellId in range(nCells):
        if cellCentered[cellId] != 0:
            for k in range(3):
                centers[3 * n + k] = centers[3 * cellId + k]
            n += 1
    numPoints[0] = Int64(n)


# ===================================================================
# Filters/Core/vtkMassProperties.cxx
# ===================================================================


# The 12 outputs land in `result` in the order of the members of
# vtkMassProperties.h.  `result` must therefore be at least 173 doubles: the
# per-triangle scratch below is carved out of the same buffer starting at
# `result + 128`, and the fifteen 3-vectors run to `result + 172`.
def MassProperties(
    pts: PtrF,
    polysConn: PtrI,
    polysOff: PtrI,
    numPolys: Int,
    numPts: Int,
    result: PtrF,
):
    var x = result + 128
    var y = result + 131
    var z = result + 134
    var i = result + 137
    var j = result + 140
    var k = result + 143
    var u = result + 146
    var absu = result + 149
    var xp = result + 152
    var vol = result + 155
    var kxyz = result + 158
    var munc = result + 161
    var ii = result + 164
    var jj = result + 167
    var kk = result + 170

    var surfacearea = 0.0
    var volumeproj = 0.0
    var mincellarea = VTK_DOUBLE_MAX
    var maxcellarea = 0.0
    var wxyz = 0.0
    var wxy = 0.0
    var wxz = 0.0
    var wyz = 0.0
    for idx in range(3):
        munc[idx] = 0.0
        vol[idx] = 0.0
        kxyz[idx] = 0.0

    for cellId in range(numPolys):
        var npts = GetNumberOfIds(polysOff, cellId)
        if PolyCellType(npts) != VTK_TRIANGLE:
            continue
        # store current vertex (x,y,z) coordinates
        for idx in range(npts):
            var p = Int(polysConn[Int(polysOff[cellId]) + idx])
            x[idx] = pts[3 * p]
            y[idx] = pts[3 * p + 1]
            z[idx] = pts[3 * p + 2]

        # get i j k vectors
        i[0] = (x[1] - x[0])
        j[0] = (y[1] - y[0])
        k[0] = (z[1] - z[0])
        i[1] = (x[2] - x[0])
        j[1] = (y[2] - y[0])
        k[1] = (z[2] - z[0])
        i[2] = (x[2] - x[1])
        j[2] = (y[2] - y[1])
        k[2] = (z[2] - z[1])

        # cross product between two vectors, to determine normal vector
        u[0] = (j[0] * k[1] - k[0] * j[1])
        u[1] = (k[0] * i[1] - i[0] * k[1])
        u[2] = (i[0] * j[1] - j[0] * i[1])

        # normalize normal vector to 1
        var length = Norm(u)
        if length != 0.0:
            u[0] /= length
            u[1] /= length
            u[2] /= length
        else:
            u[0] = 0.0
            u[1] = 0.0
            u[2] = 0.0

        # determine max unit normal component
        absu[0] = abs(u[0])
        absu[1] = abs(u[1])
        absu[2] = abs(u[2])

        if (absu[0] > absu[1]) and (absu[0] > absu[2]):
            munc[0] += 1.0
        elif (absu[1] > absu[0]) and (absu[1] > absu[2]):
            munc[1] += 1.0
        elif (absu[2] > absu[0]) and (absu[2] > absu[1]):
            munc[2] += 1.0
        elif (absu[0] == absu[1]) and (absu[0] == absu[2]):
            wxyz += 1.0
        elif (absu[0] == absu[1]) and (absu[0] > absu[2]):
            wxy += 1.0
        elif (absu[0] == absu[2]) and (absu[0] > absu[1]):
            wxz += 1.0
        elif (absu[1] == absu[2]) and (absu[0] < absu[2]):
            wyz += 1.0

        # This is reduced to ...
        ii[0] = i[0] * i[0]
        ii[1] = i[1] * i[1]
        ii[2] = i[2] * i[2]
        jj[0] = j[0] * j[0]
        jj[1] = j[1] * j[1]
        jj[2] = j[2] * j[2]
        kk[0] = k[0] * k[0]
        kk[1] = k[1] * k[1]
        kk[2] = k[2] * k[2]

        # area of a triangle
        var a = sqrt(ii[1] + jj[1] + kk[1])
        var b = sqrt(ii[0] + jj[0] + kk[0])
        var c = sqrt(ii[2] + jj[2] + kk[2])
        var s = 0.5 * (a + b + c)
        var area = sqrt(abs(s * (s - a) * (s - b) * (s - c)))
        surfacearea += area
        if area < mincellarea:
            mincellarea = area
        if area > maxcellarea:
            maxcellarea = area

        # volume elements
        var zavg = (z[0] + z[1] + z[2]) / 3.0
        var yavg = (y[0] + y[1] + y[2]) / 3.0
        var xavg = (x[0] + x[1] + x[2]) / 3.0

        vol[2] += (area * u[2] * zavg)
        vol[1] += (area * u[1] * yavg)
        vol[0] += (area * u[0] * xavg)

        # V  =  (z1+z2+z3)(x1y2-x2y1+x2y3-x3y2+x3y1-x1y3)/6
        Cross(x, y, xp)
        volumeproj += zavg * (xp[0] + xp[1] + xp[2]) / 2

    # Surface Area
    result[0] = surfacearea
    result[1] = mincellarea
    result[2] = maxcellarea

    # Weighting factors in Discrete Divergence theorem for volume calculation.
    var nc = Float64(numPolys)
    kxyz[0] = (munc[0] + (wxyz / 3.0) + ((wxy + wxz) / 2.0)) / nc
    kxyz[1] = (munc[1] + (wxyz / 3.0) + ((wxy + wyz) / 2.0)) / nc
    kxyz[2] = (munc[2] + (wxyz / 3.0) + ((wxz + wyz) / 2.0)) / nc
    result[3] = vol[0]
    result[4] = vol[1]
    result[5] = vol[2]
    result[6] = kxyz[0]
    result[7] = kxyz[1]
    result[8] = kxyz[2]
    result[9] = (kxyz[0] * vol[0] + kxyz[1] * vol[1] + kxyz[2] * vol[2])
    result[9] = abs(result[9])
    result[10] = volumeproj
    result[11] = (sqrt(surfacearea) / cbrt(result[9])) / 2.199085233


# ===================================================================
# Filters/Core/vtkFeatureEdges.cxx
# ===================================================================


def FeatureEdges(
    pts: PtrF,
    polysConn: PtrI,
    polysOff: PtrI,
    numPolys: Int,
    numPts: Int,
    featureAngle: Float64,
    boundaryEdges: Bool,
    featureEdgesOn: Bool,
    nonManifoldEdges: Bool,
    manifoldEdges: Bool,
    coloring: Bool,
    newPts: PtrF,
    newLinesConn: PtrI,
    newLinesOff: PtrI,
    edgeTypes: PtrF,
    polyNormals: PtrF,
    outSrc: PtrI,
    counts: PtrI,
    scratchF: PtrF,
    scratchI: PtrI,
):
    var sI = scratchI
    var numEdges = 0
    for c in range(numPolys):
        numEdges += GetNumberOfIds(polysOff, c)
    var slotCap = 4
    while slotCap < 2 * numEdges + 2:
        slotCap *= 2

    # edge table, indexed out of the caller's int pool
    var slots = sI
    sI += slotCap
    var slotKeys = sI
    sI += slotCap
    var edgeCell = sI
    sI += numEdges
    var edgeNext = sI
    sI += numEdges
    # one cell can contribute more than one entry to an edge (a degenerate cell
    # repeats a point id), so the neighbour list is sized like vtkIdList grows
    var neighbors = sI
    sI += numEdges
    # `Locator->InsertUniquePoint`: the output line endpoints are renumbered
    # into a fresh, compact point array.
    var pointMap = sI
    sI += numPts
    for i in range(numPts):
        pointMap[i] = -1
    BuildEdgeTable(
        polysConn, polysOff, numPolys, numPts, slots, slotKeys, slotCap, edgeCell, edgeNext
    )

    var n = scratchF
    var w = scratchF + 3

    #  Initial pass to compute polygon normals without effects of neighbors
    if featureEdgesOn:
        for cellId in range(numPolys):
            var npts = GetNumberOfIds(polysOff, cellId)
            ComputeNormal(pts, npts, polysConn + Int(polysOff[cellId]), 1, n, w)
            for k in range(3):
                polyNormals[3 * cellId + k] = n[k]
    var cosAngle = 0.0
    if featureEdgesOn:
        cosAngle = cos(RadiansFromDegrees(featureAngle))

    var numBEdges = 0
    var numNonManifoldEdges = 0
    var numFedges = 0
    var numManifoldEdges = 0
    var scalar = 0.0
    var newPtsCount = 0
    var newLines = 0

    for newCellId in range(numPolys):
        var npts = GetNumberOfIds(polysOff, newCellId)
        for i in range(npts):
            var p1 = Int(polysConn[Int(polysOff[newCellId]) + i])
            var p2 = Int(polysConn[Int(polysOff[newCellId]) + (i + 1) % npts])
            var numNei = GetCellEdgeNeighbors(
                slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, newCellId, neighbors
            )
            var numNeiWithoutGhosts = numNei
            var firstNeighbor = 0

            if boundaryEdges and numNeiWithoutGhosts < 1:
                numBEdges += 1
                scalar = 0.0

            elif nonManifoldEdges and numNeiWithoutGhosts > 1:
                # check to make sure that this edge hasn't been created before
                var j = 0
                while j < numNei:
                    if Int(neighbors[j]) < newCellId:
                        break
                    j += 1
                if j >= numNeiWithoutGhosts:
                    numNonManifoldEdges += 1
                    scalar = 0.222222
                else:
                    continue
            elif (
                featureEdgesOn
                and numNeiWithoutGhosts == 1
                and Int(neighbors[firstNeighbor]) > newCellId
            ):
                var nei = Int(neighbors[firstNeighbor])
                if Dot(polyNormals + 3 * nei, polyNormals + 3 * newCellId) <= cosAngle:
                    numFedges += 1
                    scalar = 0.444444
                else:
                    continue
            elif (
                manifoldEdges
                and numNeiWithoutGhosts == 1
                and Int(neighbors[firstNeighbor]) > newCellId
            ):
                numManifoldEdges += 1
                scalar = 0.666667
            else:
                continue

            # Add edge to output
            if pointMap[p1] == -1:
                pointMap[p1] = Int64(newPtsCount)
                for k in range(3):
                    newPts[3 * newPtsCount + k] = pts[3 * p1 + k]
                newPtsCount += 1
            if pointMap[p2] == -1:
                pointMap[p2] = Int64(newPtsCount)
                for k in range(3):
                    newPts[3 * newPtsCount + k] = pts[3 * p2 + k]
                newPtsCount += 1
            var lineIds0 = Int(pointMap[p1])
            var lineIds1 = Int(pointMap[p2])

            var cellId = newLines // 2
            newLinesConn[newLines] = Int64((lineIds0))
            newLinesConn[newLines + 1] = Int64((lineIds1))
            newLinesOff[cellId + 1] = Int64((newLines + 2))
            outSrc[cellId] = Int64((newCellId))
            if coloring:
                edgeTypes[cellId] = scalar
            newLines += 2

    counts[0] = Int64(newPtsCount)
    counts[1] = Int64(newLines // 2)
    counts[2] = Int64(numBEdges)
    counts[3] = Int64(numNonManifoldEdges)
    counts[4] = Int64(numFedges)
    counts[5] = Int64(numManifoldEdges)


# ===================================================================
# Filters/Core/vtkPolyDataNormals.cxx
# ===================================================================


# vtkPolyDataNormals::GetCellNormals
def GetCellNormals(
    pts: PtrF,
    numVerts: Int,
    numLines: Int,
    polysConn: PtrI,
    polysOff: PtrI,
    numPolys: Int,
    numStrips: Int,
    cellNormals: PtrF,
    scratchF: PtrF,
):
    var n = scratchF
    var w = scratchF + 3
    # Set default value for vertices and lines cell normals
    # normals do not have meaningful values, we set them to X
    var offsetCells = 0
    while offsetCells < numVerts + numLines:
        cellNormals[3 * offsetCells] = 1.0
        cellNormals[3 * offsetCells + 1] = 0.0
        cellNormals[3 * offsetCells + 2] = 0.0
        offsetCells += 1

    # Compute Cell Normals of polys
    for polyId in range(numPolys):
        var npts = GetNumberOfIds(polysOff, polyId)
        ComputeNormal(pts, npts, polysConn + Int(polysOff[polyId]), 1, n, w)
        for k in range(3):
            cellNormals[3 * (offsetCells + polyId) + k] = n[k]

    # Set default value for strip cell normals
    offsetCells += numPolys
    var cellId = 0
    while cellId < numStrips:
        cellNormals[3 * (offsetCells + cellId)] = 1.0
        cellNormals[3 * (offsetCells + cellId) + 1] = 0.0
        cellNormals[3 * (offsetCells + cellId) + 2] = 0.0
        cellId += 1


# vtkPolyDataNormals::GetPointNormals
def GetPointNormals(
    pts: PtrF,
    polysConn: PtrI,
    polysOff: PtrI,
    numPolys: Int,
    numPts: Int,
    cellNormals: PtrF,
    pointNormals: PtrF,
    flipDirection: Float64,
    linkOffsets: PtrI,
    linkCells: PtrI,
    cursor: PtrI,
):
    BuildPointLinks(polysConn, polysOff, numPolys, numPts, linkOffsets, linkCells, cursor)
    for pointId in range(numPts):
        var pointNormal = pointNormals + 3 * pointId
        # Initialize point normals
        pointNormal[0] = 0.0
        pointNormal[1] = 0.0
        pointNormal[2] = 0.0
        # Compute Point Normals
        var nCells = Int(linkOffsets[pointId + 1] - linkOffsets[pointId])
        for i in range(nCells):
            Add(pointNormal, cellNormals + 3 * Int(linkCells[Int(linkOffsets[pointId]) + i]), pointNormal)
        # Normalize normals
        var length = Norm(pointNormal) * flipDirection
        if length != 0.0:
            MultiplyScalar(pointNormal, 1.0 / length)


# ===================================================================
# Filters/Core/vtkPointDataToCellData.cxx
# ===================================================================


# PointDataToCellDataFunctor: non-categorical -> simply average the data.
def PointDataToCellData(
    scalars: PtrF,
    ncomp: Int,
    conn: PtrI,
    offsets: PtrI,
    nCells: Int,
    dst: PtrF,
    hasData: PtrT,
):
    for cellId in range(nCells):
        var npts = GetNumberOfIds(offsets, cellId)
        if npts == 0:
            continue
        for c in range(ncomp):
            var acc = 0.0
            for i in range(npts):
                acc += scalars[ncomp * Int(conn[Int(offsets[cellId]) + i]) + c]
            dst[ncomp * cellId + c] = acc / Float64(npts)
        hasData[cellId] = 1


# PointDataToCellDataCategoricalFunctor: majority value wins.
def PointDataToCellDataCategorical(
    scalars: PtrF,
    conn: PtrI,
    offsets: PtrI,
    nCells: Int,
    dst: PtrF,
    hasData: PtrT,
    bins: PtrF,
) -> Bool:
    for cellId in range(nCells):
        var npts = GetNumberOfIds(offsets, cellId)
        if npts == 0:
            continue
        # Populate a histogram from the scalar values at each point, and then
        # select the bin with the most elements.
        for i in range(npts + 1):
            bins[3 * i] = -1.0
            bins[3 * i + 1] = 1.0
            bins[3 * i + 2] = VTK_DOUBLE_MAX
        var counter = 0
        for ptId in range(npts):
            var pointId = Int(conn[Int(offsets[cellId]) + ptId])
            bins[3 * counter] = Float64(pointId)
            bins[3 * counter + 2] = scalars[pointId]
            counter += 1
        # `Arrays.Copy(index, cellId)` writes the VALUE stored at that point
        dst[cellId] = scalars[Int(IndexOfLargestBin(bins, counter, npts))]
        hasData[cellId] = 1
    return True


# Sort the histogram bins by value, effectively grouping like bins
def IndexOfLargestBin(bins: PtrF, counter: Int, npts: Int) -> Float64:
    # If there is only one datapoint, return its index
    if counter == 1:
        return bins[0]
    var count = counter
    # insertion sort on (value) -- vtkPriorityQueue is not needed, the bins are
    # grouped by an adjacent sweep exactly as upstream
    for i in range(1, count):
        var v = bins[3 * i + 2]
        var idx = bins[3 * i]
        var c = bins[3 * i + 1]
        var j = i - 1
        while j >= 0 and bins[3 * j + 2] > v:
            bins[3 * (j + 1)] = bins[3 * j]
            bins[3 * (j + 1) + 1] = bins[3 * j + 1]
            bins[3 * (j + 1) + 2] = bins[3 * j + 2]
            j -= 1
        bins[3 * (j + 1)] = idx
        bins[3 * (j + 1) + 1] = c
        bins[3 * (j + 1) + 2] = v

    # Perform a single sweep, comparing adjacent bins.
    var it2 = 1
    var it1 = 0
    while it2 < count and bins[3 * it2] != -1.0:
        # If the adjacent bins are close enough to be merged...
        if abs(bins[3 * it1 + 2] - bins[3 * it2 + 2]) < 1.0e-6:
            #...then increment the count of the first bin. We need not worry
            # about the count of the second bin, since it will remain 1 and
            # will therefore not be in contention for the largest bin.
            bins[3 * it1 + 1] += 1.0
        else:
            #...otherwise, move on to the next potential set of merges.
            it1 = it2
        it2 += 1

    # Finally, return the index of the element with the largest count. If there
    # is more than one, the index of the element with the smallest value is
    # consistently chosen.
    # BinCountCmp: fewer counts is "less"; on a tie the smaller value is
    # "less", so max_element is the largest count and then the largest value.
    var best = 0
    for i in range(1, it2):
        var ci = bins[3 * i + 1]
        var cb = bins[3 * best + 1]
        if ci > cb or (ci == cb and bins[3 * i + 2] > bins[3 * best + 2]):
            best = i
    return bins[3 * best]


# ===================================================================
# Filters/Core/vtkCellDataToPointData.cxx -- the `Spread` helper
# ===================================================================


# UnstructuredDataCD2PD: for every point, average the cell values of the cells
# that touch it; a point with no cells gets the array's null value.
def CellDataToPointData(
    cellScalars: PtrF,
    ncomp: Int,
    conn: PtrI,
    offsets: PtrI,
    nCells: Int,
    numPts: Int,
    dst: PtrF,
    linkOffsets: PtrI,
    linkCells: PtrI,
    cursor: PtrI,
):
    BuildPointLinks(conn, offsets, nCells, numPts, linkOffsets, linkCells, cursor)
    for pointId in range(numPts):
        var ncells = Int(linkOffsets[pointId + 1] - linkOffsets[pointId])
        if ncells > 0:
            for c in range(ncomp):
                var acc = 0.0
                for k in range(ncells):
                    acc += cellScalars[ncomp * Int(linkCells[Int(linkOffsets[pointId]) + k]) + c]
                dst[ncomp * pointId + c] = acc / Float64(ncells)
        else:
            # AssignNullValue
            for c in range(ncomp):
                dst[ncomp * pointId + c] = 0.0


# ===================================================================
# Common/ComputationalGeometry/vtkCardinalSpline.cxx
# ===================================================================


# `vtkSpline::FindIndex`: bisection method for speed.
def FindIndex(intervals: PtrF, size: Int, t: Float64) -> Int:
    var index = 0
    if size > 2:  # bisection method for speed
        var rightIdx = size - 1
        var centerIdx = rightIdx - size // 2
        var converged = 0
        while not converged:
            if (intervals[index] <= t) and (t <= intervals[centerIdx]):
                rightIdx = centerIdx
            else:  # if ( Intervals[centerIdx] < t && t <= Intervals[rightIdx] )
                index = centerIdx
            if (index + 1) == rightIdx:
                converged = 1
            else:
                centerIdx = index + (rightIdx - index) // 2
        # while not converged
    return index


# `vtkSpline::ComputeLeftDerivative` / `vtkSpline::ComputeRightDerivative`.
# Both read the piecewise function's interleaved (t, x) buffer, so the value
# is the gap to the neighbouring interval, not a slope.
def ComputeLeftDerivative(intervals: PtrF, size: Int) -> Float64:
    if size < 2:
        return 0.0
    return intervals[1] - intervals[0]


def ComputeRightDerivative(intervals: PtrF, size: Int) -> Float64:
    if size < 2:
        return 0.0
    return intervals[size - 1] - intervals[size - 2]


# vtkCardinalSpline::Fit1D -- the spline is open.
def Fit1D(
    x: PtrF,
    y: PtrF,
    work: PtrF,
    coefficients: PtrF,
    size: Int,
    leftConstraint: Int,
    leftValue: Float64,
    rightConstraint: Int,
    rightValue: Float64,
) -> Float64:
    var leftDerivative = ComputeLeftDerivative(x, size)
    var rightDerivative = ComputeRightDerivative(x, size)
    var b = 0.0
    var xlk = 0.0
    var xlkp = 0.0
    var k = 0

    # develop constraint at leftmost point.
    if leftConstraint == 0:
        # desired slope at leftmost point is derivative from two points
        coefficients[4 * 0 + 1] = 1.0
        coefficients[4 * 0 + 2] = 0.0
        work[0] = leftDerivative
    elif leftConstraint == 1:
        # desired slope at leftmost point is leftValue.
        coefficients[4 * 0 + 1] = 1.0
        coefficients[4 * 0 + 2] = 0.0
        work[0] = leftValue
    elif leftConstraint == 2:
        # desired second derivative at leftmost point is leftValue.
        coefficients[4 * 0 + 1] = 2.0
        coefficients[4 * 0 + 2] = 1.0
        work[0] = 3.0 * ((y[1] - y[0]) / (x[1] - x[0])) - 0.5 * (x[1] - x[0]) * leftValue
    else:
        # desired second derivative at leftmost point is leftValue times second
        # derivative at first interior point.
        coefficients[4 * 0 + 1] = 2.0
        coefficients[4 * 0 + 2] = 4.0 * ((0.5 + leftValue) / (2.0 + leftValue))
        work[0] = (
            6.0 * ((1.0 + leftValue) / (2.0 + leftValue)) * ((y[1] - y[0]) / (x[1] - x[0]))
        )

    # develop body of band matrix.
    k = 1
    while k < size - 1:
        xlk = x[k] - x[k - 1]
        xlkp = x[k + 1] - x[k]
        coefficients[4 * k] = xlkp
        coefficients[4 * k + 1] = 2.0 * (xlkp + xlk)
        coefficients[4 * k + 2] = xlk
        work[k] = 3.0 * (
            ((xlkp * (y[k] - y[k - 1])) / xlk) + ((xlk * (y[k + 1] - y[k])) / xlkp)
        )
        k += 1

    # develop constraint at rightmost point.
    if rightConstraint == 0:
        # desired slope at leftmost point is derivative from two points
        coefficients[4 * (size - 1)] = 0.0
        coefficients[4 * (size - 1) + 1] = 1.0
        work[size - 1] = rightDerivative
    elif rightConstraint == 1:
        # desired slope at rightmost point is rightValue
        coefficients[4 * (size - 1)] = 0.0
        coefficients[4 * (size - 1) + 1] = 1.0
        work[size - 1] = rightValue
    elif rightConstraint == 2:
        # desired second derivative at rightmost point is rightValue.
        coefficients[4 * (size - 1)] = 1.0
        coefficients[4 * (size - 1) + 1] = 2.0
        work[size - 1] = 3.0 * ((y[size - 1] - y[size - 2]) / (x[size - 1] - x[size - 2])) + (
            0.5 * (x[size - 1] - x[size - 2]) * rightValue
        )
    else:
        # desired second derivative at rightmost point is rightValue times second
        # derivative at last interior point.
        coefficients[4 * (size - 1)] = 4.0 * ((0.5 + rightValue) / (2.0 + rightValue))
        coefficients[4 * (size - 1) + 1] = 2.0
        work[size - 1] = (
            6.0
            * ((1.0 + rightValue) / (2.0 + rightValue))
            * ((y[size - 1] - y[size - 2]) / (x[size - 1] - x[size - 2]))
        )

    # solve resulting set of equations.
    coefficients[2] = coefficients[2] / coefficients[1]
    work[0] = work[0] / coefficients[1]
    coefficients[4 * (size - 1) + 2] = 0.0

    k = 1
    while k < size:
        coefficients[4 * k + 1] = coefficients[4 * k + 1] - (
            coefficients[4 * k] * coefficients[4 * (k - 1) + 2]
        )
        coefficients[4 * k + 2] = coefficients[4 * k + 2] / coefficients[4 * k + 1]
        work[k] = (work[k] - (coefficients[4 * k] * work[k - 1])) / coefficients[4 * k + 1]
        k += 1

    k = size - 2
    while k >= 0:
        work[k] = work[k] - (coefficients[4 * k + 2] * work[k + 1])
        k -= 1

    # the column vector work now contains the first derivative of the spline
    # function at each joint.  Compute the coefficients of the cubic between
    # each pair of joints.
    k = 0
    while k < size - 1:
        b = x[k + 1] - x[k]
        coefficients[4 * k] = y[k]
        coefficients[4 * k + 1] = work[k]
        coefficients[4 * k + 2] = (3.0 * (y[k + 1] - y[k])) / (b * b) - (work[k + 1] + 2.0 * work[k]) / b
        coefficients[4 * k + 3] = (
            (2.0 * (y[k] - y[k + 1])) / (b * b * b) + (work[k + 1] + work[k]) / (b * b)
        )
        k += 1

    # the coefficients of a fictitious nth cubic are evaluated.  This may
    # simplify algorithms which include both end points.
    coefficients[4 * (size - 1)] = y[size - 1]
    coefficients[4 * (size - 1) + 1] = work[size - 1]
    coefficients[4 * (size - 1) + 2] = (
        coefficients[4 * (size - 2) + 2] + 3.0 * coefficients[4 * (size - 2) + 3] * b
    )
    coefficients[4 * (size - 1) + 3] = coefficients[4 * (size - 2) + 3]
    return b


# vtkCardinalSpline::FitClosed1D
def FitClosed1D(x: PtrF, y: PtrF, work: PtrF, coefficients: PtrF, size: Int) -> Float64:
    var b = 0.0
    var xlk = 0.0
    var xlkp = 0.0
    var k = 0
    var N = size - 1

    # develop body of band matrix.
    k = 1
    while k < N:
        xlk = x[k] - x[k - 1]
        xlkp = x[k + 1] - x[k]
        coefficients[4 * k] = xlkp
        coefficients[4 * k + 1] = 2.0 * (xlkp + xlk)
        coefficients[4 * k + 2] = xlk
        work[k] = 3.0 * (
            ((xlkp * (y[k] - y[k - 1])) / xlk) + ((xlk * (y[k + 1] - y[k])) / xlkp)
        )
        k += 1

    xlk = x[N] - x[N - 1]
    xlkp = x[1] - x[0]
    var aN = xlkp
    coefficients[4 * N] = xlkp
    var bN = 2.0 * (xlkp + xlk)
    coefficients[4 * N + 1] = bN
    var cN = xlk
    coefficients[4 * N + 2] = xlk
    var dN = 3.0 * (
        ((xlkp * (y[N] - y[N - 1])) / xlk) + ((xlk * (y[1] - y[0])) / xlkp)
    )
    work[N] = dN

    # solve resulting set of equations.
    coefficients[2] = 0.0
    work[0] = 0.0
    coefficients[3] = 1.0

    k = 1
    while k <= N:
        coefficients[4 * k + 1] = coefficients[4 * k + 1] - (
            coefficients[4 * k] * coefficients[4 * (k - 1) + 2]
        )
        coefficients[4 * k + 2] = coefficients[4 * k + 2] / coefficients[4 * k + 1]
        work[k] = (work[k] - (coefficients[4 * k] * work[k - 1])) / coefficients[4 * k + 1]
        coefficients[4 * k + 3] = (-1.0 * coefficients[4 * k] * coefficients[4 * (k - 1) + 3]) / (
            coefficients[4 * k + 1]
        )
        k += 1

    coefficients[4 * N] = 1.0
    coefficients[4 * N + 1] = 0.0

    k = N - 1
    while k > 0:
        coefficients[4 * k] = coefficients[4 * k + 3] - coefficients[4 * k + 2] * coefficients[
            4 * (k + 1)
        ]
        coefficients[4 * k + 1] = work[k] - coefficients[4 * k + 2] * coefficients[4 * (k + 1) + 1]
        k -= 1

    work[0] = work[N] = (dN - cN * coefficients[5] - aN * coefficients[4 * (N - 1) + 1]) / (
        bN + cN * coefficients[4] + aN * coefficients[4 * (N - 1)]
    )

    k = 1
    while k < N:
        work[k] = coefficients[4 * k] * work[N] + coefficients[4 * k + 1]
        k += 1

    # the column vector work now contains the first derivative of the spline
    # function at each joint.
    k = 0
    while k < N:
        b = x[k + 1] - x[k]
        coefficients[4 * k] = y[k]
        coefficients[4 * k + 1] = work[k]
        coefficients[4 * k + 2] = (3.0 * (y[k + 1] - y[k])) / (b * b) - (work[k + 1] + 2.0 * work[k]) / b
        coefficients[4 * k + 3] = (
            (2.0 * (y[k] - y[k + 1])) / (b * b * b) + (work[k + 1] + work[k]) / (b * b)
        )
        k += 1

    # the coefficients of a fictitious nth cubic are the same as the
    # coefficients in the first interval
    coefficients[4 * N] = y[N]
    coefficients[4 * N + 1] = work[N]
    coefficients[4 * N + 2] = coefficients[2]
    coefficients[4 * N + 3] = coefficients[3]
    return b


# vtkCardinalSpline::Compute
def CardinalSplineCompute(
    ts: PtrF,
    ys: PtrF,
    size: Int,
    closed: Bool,
    parametricRange: PtrF,
    intervals: PtrF,
    work: PtrF,
    dependent: PtrF,
    coefficients: PtrF,
    leftConstraint: Int,
    leftValue: Float64,
    rightConstraint: Int,
    rightValue: Float64,
) -> Int:
    var n = size
    if size < 2:
        return 0
    if not closed:
        for i in range(size):
            intervals[i] = ts[2 * i]
        # get the dependent variable values; `xs` is the buffer offset by one,
        # so this is the value half of each (t, x) pair
        for j in range(size):
            dependent[j] = ys[2 * j + 1]
        Fit1D(
            intervals,
            dependent,
            work,
            coefficients,
            size,
            leftConstraint,
            leftValue,
            rightConstraint,
            rightValue,
        )
    else:
        # add extra "fictitious" point to close loop
        n = size + 1
        for i in range(size):
            intervals[i] = ts[2 * i]
        if parametricRange[0] != parametricRange[1]:
            intervals[size] = parametricRange[1]
        else:
            intervals[size] = intervals[size - 1] + 1.0
        for j in range(size):
            dependent[j] = ys[2 * j + 1]
        dependent[size] = ys[1]
        FitClosed1D(intervals, dependent, work, coefficients, n)
    return n


# vtkCardinalSpline::Evaluate
def CardinalSplineEvaluate(intervals: PtrF, coefficients: PtrF, size: Int, t: Float64) -> Float64:
    var tt = t
    # make sure we have at least 2 points
    if size < 2:
        return 0.0
    # clamp the function at both ends
    if tt < intervals[0]:
        tt = intervals[0]
    if tt > intervals[size - 1]:
        tt = intervals[size - 1]
    # find pointer to cubic spline coefficient using bisection method
    var index = FindIndex(intervals, size, tt)
    # calculate offset within interval
    tt = (tt - intervals[index])
    # evaluate intervals value y
    return (
        tt
        * (tt * (tt * coefficients[index * 4 + 3] + coefficients[index * 4 + 2]) + coefficients[
            index * 4 + 1
        ])
        + coefficients[index * 4]
    )


# ===================================================================
# Filters/Modeling/vtkLoopSubdivisionFilter.cxx
# ===================================================================

# `static constexpr double LoopWeights[4]` in vtkLoopSubdivisionFilter.cxx
def LoopWeights(i: Int) -> Float64:
    if i == 0 or i == 1:
        return 0.375
    return 0.125


# vtkApproximatingSubdivisionFilter::InterpolatePosition
def InterpolatePosition(
    inputPts: PtrF, stencil: PtrI, nStencil: Int, weights: PtrF, outputPts: PtrF, numOut: PtrI, w: PtrF
) -> Int:
    var x = w
    x[0] = 0.0
    x[1] = 0.0
    x[2] = 0.0
    for i in range(nStencil):
        var s = Int(stencil[i])
        for j in range(3):
            x[j] += inputPts[3 * s + j] * weights[i]
    var newId = Int(numOut[0])
    for j in range(3):
        outputPts[3 * newId + j] = x[j]
    numOut[0] = Int64(newId + 1)
    return newId


# vtkLoopSubdivisionFilter::GenerateEvenStencil
def GenerateEvenStencil(
    p1: Int,
    polysConn: PtrI,
    polysOff: PtrI,
    numPts: Int,
    linkOffsets: PtrI,
    linkCells: PtrI,
    slots: PtrI,
    slotKeys: PtrI,
    slotCap: Int,
    edgeCell: PtrI,
    edgeNext: PtrI,
    neighbors: PtrI,
    stencil: PtrI,
    nStencil: PtrI,
    weights: PtrF,
) -> Int:
    # Get the cells that use this point
    var numCellsInLoop = Int(linkOffsets[p1 + 1] - linkOffsets[p1])
    if numCellsInLoop < 1:
        nStencil[0] = 0
        return 0
    # Find an edge to start with that contains p1
    var firstCell = Int(linkCells[Int(linkOffsets[p1])])
    var p2 = Int(polysConn[Int(polysOff[firstCell])])
    var i = 1
    while p1 == p2:
        p2 = Int(polysConn[Int(polysOff[firstCell]) + i])
        i += 1
    var numNei = GetCellEdgeNeighbors(
        slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, -1, neighbors
    )

    var nextCell = Int(neighbors[0])
    var bp2 = -1
    var bp1 = p2
    var startCell = -1
    if numNei != 1:
        startCell = Int(neighbors[1])

    nStencil[0] = 1
    stencil[0] = Int64(p2)

    # walk around the loop counter-clockwise and get cells
    var j = 0
    while j < numCellsInLoop:
        var p = -1
        for k in range(3):
            var q = Int(polysConn[Int(polysOff[nextCell]) + k])
            if q != p1 and q != p2:
                p = q
                break
        p2 = p
        stencil[nStencil[0]] = Int64(p2)
        nStencil[0] += 1
        var numNei2 = GetCellEdgeNeighbors(
            slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, nextCell, neighbors
        )
        if numNei2 != 1:
            bp2 = p2
            j += 1
            break
        nextCell = Int(neighbors[0])
        j += 1

    # now walk around the other way. this will only happen if there is a
    # boundary cell left that we have not visited
    nextCell = startCell
    p2 = bp1
    while j < numCellsInLoop and startCell != -1:
        var p = -1
        for k in range(3):
            var q = Int(polysConn[Int(polysOff[nextCell]) + k])
            if q != p1 and q != p2:
                p = q
                break
        p2 = p
        stencil[nStencil[0]] = Int64(p2)
        nStencil[0] += 1
        var numNei2 = GetCellEdgeNeighbors(
            slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, nextCell, neighbors
        )
        if numNei2 != 1:
            bp1 = p2
            break
        nextCell = Int(neighbors[0])
        j += 1

    if bp2 != -1:  # boundary edge
        nStencil[0] = 3
        stencil[0] = Int64(bp2)
        stencil[1] = Int64(bp1)
        stencil[2] = Int64(p1)
        weights[0] = 0.125
        weights[1] = 0.125
        weights[2] = 0.75
    else:
        var K = Int(nStencil[0])
        # Remove last id. It's a duplicate of the first
        K -= 1
        var beta = 0.0
        if K > 3:
            # Generate weights
            var cosSQ = 0.375 + 0.25 * cos(2.0 * VTK_PI / Float64(K))
            cosSQ = cosSQ * cosSQ
            beta = (0.625 - cosSQ) / Float64(K)
        else:
            beta = 3.0 / 16.0
        for k in range(K):
            weights[k] = beta
        weights[K] = 1.0 - Float64(K) * beta
        stencil[K] = Int64(p1)
    return 1


# vtkLoopSubdivisionFilter::GenerateOddStencil
def GenerateOddStencil(
    p1: Int,
    p2: Int,
    polysConn: PtrI,
    polysOff: PtrI,
    slots: PtrI,
    slotKeys: PtrI,
    slotCap: Int,
    edgeCell: PtrI,
    edgeNext: PtrI,
    numPts: Int,
    stencil: PtrI,
    weights: PtrF,
    neighbors: PtrI,
):
    var numNei = GetCellEdgeNeighbors(
        slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, -1, neighbors
    )
    var cell0 = Int(neighbors[0])
    var cell1 = Int(neighbors[1])

    var p3 = 0
    var p4 = 0
    for k in range(3):
        var q = Int(polysConn[Int(polysOff[cell0]) + k])
        if q != p1 and q != p2:
            p3 = q
            break
    for k in range(3):
        var q = Int(polysConn[Int(polysOff[cell1]) + k])
        if q != p1 and q != p2:
            p4 = q
            break

    stencil[0] = Int64(p1)
    stencil[1] = Int64(p2)
    stencil[2] = Int64(p3)
    stencil[3] = Int64(p4)

    for i in range(4):
        weights[i] = LoopWeights(i)


# vtkApproximatingSubdivisionFilter::FindEdge
def FindEdge(
    polysConn: PtrI,
    polysOff: PtrI,
    cellId: Int,
    p1: Int,
    p2: Int,
    edgeData: PtrI,
    slots: PtrI,
    slotKeys: PtrI,
    slotCap: Int,
    edgeCell: PtrI,
    edgeNext: PtrI,
    numPts: Int,
    neighbors: PtrI,
) -> Int:
    var edgeId = 0
    var currentCellId = 0
    # get all the cells that use the edge (except for cellId)
    var numNei = GetCellEdgeNeighbors(
        slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, cellId, neighbors
    )
    # find the edge that has the point we are looking for
    for i in range(numNei):
        currentCellId = Int(neighbors[i])
        var numEdges = GetNumberOfIds(polysOff, currentCellId)
        var tp1 = Int(polysConn[Int(polysOff[currentCellId]) + 2])
        var tp2 = Int(polysConn[Int(polysOff[currentCellId])])
        edgeId = 0
        while edgeId < numEdges:
            if (tp1 == p1 and tp2 == p2) or (tp2 == p1 and tp1 == p2):
                break
            tp1 = tp2
            tp2 = Int(polysConn[Int(polysOff[currentCellId]) + (edgeId + 1) % numEdges])
            edgeId += 1
    return Int(edgeData[3 * currentCellId + edgeId])


# vtkApproximatingSubdivisionFilter::GenerateSubdivisionCells
def GenerateSubdivisionCells(
    polysConn: PtrI,
    polysOff: PtrI,
    numCells: Int,
    edgeData: PtrI,
    oconn: PtrI,
    ooffsets: PtrI,
    outSrc: PtrI,
) -> Int:
    var newId = 0
    var conn = 0
    ooffsets[0] = 0
    for cellId in range(numCells):
        if PolyCellType(GetNumberOfIds(polysOff, cellId)) != VTK_TRIANGLE:
            continue
        var pts = polysConn + Int(polysOff[cellId])
        var edgePts = edgeData + 3 * cellId
        var id = 0
        oconn[conn] = pts[0]
        oconn[conn + 1] = edgePts[1]
        oconn[conn + 2] = edgePts[0]
        ooffsets[newId + 1] = Int64((conn + 3))
        outSrc[newId] = Int64((cellId))
        conn += 3
        newId += 1

        oconn[conn] = edgePts[1]
        oconn[conn + 1] = pts[1]
        oconn[conn + 2] = edgePts[2]
        ooffsets[newId + 1] = Int64((conn + 3))
        outSrc[newId] = Int64((cellId))
        conn += 3
        newId += 1

        oconn[conn] = edgePts[2]
        oconn[conn + 1] = pts[2]
        oconn[conn + 2] = edgePts[0]
        ooffsets[newId + 1] = Int64((conn + 3))
        outSrc[newId] = Int64((cellId))
        conn += 3
        newId += 1

        oconn[conn] = edgePts[1]
        oconn[conn + 1] = edgePts[2]
        oconn[conn + 2] = edgePts[0]
        ooffsets[newId + 1] = Int64((conn + 3))
        outSrc[newId] = Int64((cellId))
        conn += 3
        newId += 1
    return newId


# vtkLoopSubdivisionFilter::GenerateSubdivisionPoints
def GenerateSubdivisionPoints(
    pts: PtrF,
    polysConn: PtrI,
    polysOff: PtrI,
    numCells: Int,
    numPts: Int,
    outputPts: PtrF,
    edgeData: PtrI,
    oconn: PtrI,
    ooffsets: PtrI,
    outSrc: PtrI,
    counts: PtrI,
    scratchF: PtrF,
    scratchI: PtrI,
) -> Int:
    var sI = scratchI
    var numEdges = 0
    for c in range(numCells):
        numEdges += GetNumberOfIds(polysOff, c)
    var slotCap = 4
    while slotCap < 2 * numEdges + 2:
        slotCap *= 2

    # The edge table, then a second table holding the edges already visited.
    var slots = sI
    sI += slotCap
    var slotKeys = sI
    sI += slotCap
    var edgeCell = sI
    sI += numEdges
    var edgeNext = sI
    sI += numEdges
    var seenKeys = sI
    sI += slotCap
    var linkOffsets = sI
    sI += numPts + 1
    var linkCells = sI
    sI += numEdges
    var cursor = sI
    sI += numPts
    var cellIds = sI
    sI += numCells + 4
    var stencil = sI
    sI += numCells + 4
    var neighbors = sI
    sI += numCells + 4
    var nStencil = sI
    sI += 1
    var numOut = sI
    sI += 1
    BuildEdgeTable(
        polysConn, polysOff, numCells, numPts, slots, slotKeys, slotCap, edgeCell, edgeNext
    )
    BuildPointLinks(polysConn, polysOff, numCells, numPts, linkOffsets, linkCells, cursor)
    for i in range(slotCap):
        seenKeys[i] = -1

    numOut[0] = 0
    var weights = scratchF
    var w = scratchF + 256

    # Generate even points. these are derived from the old points
    for ptId in range(numPts):
        if (
            GenerateEvenStencil(
                ptId,
                polysConn,
                polysOff,
                numPts,
                linkOffsets,
                linkCells,
                slots,
                slotKeys,
                slotCap,
                edgeCell,
                edgeNext,
                neighbors,
                stencil,
                nStencil,
                weights,
            ) == 0
        ):
            return 0
        InterpolatePosition(pts, stencil, Int(nStencil[0]), weights, outputPts, numOut, w)

    # Generate odd points. These will be inserted into the new dataset
    for cellId in range(numCells):
        # start with one edge
        var p1 = Int(polysConn[Int(polysOff[cellId]) + 2])
        var p2 = Int(polysConn[Int(polysOff[cellId])])
        for edgeId in range(3):
            # Do we need to create a point on this edge?
            var k0 = p1
            var k1 = p2
            if k1 < k0:
                k0 = p2
                k1 = p1
            var probe = EdgeKeyHash(k0, k1, slotCap)
            var isEdge = 0
            while seenKeys[probe] != -1:
                if seenKeys[probe] == Int64(k0 * numPts + k1):
                    isEdge = 1
                    break
                probe = (probe + 1) % slotCap
            var newId = 0
            if isEdge == 0:
                seenKeys[probe] = Int64(k0 * numPts + k1)
                var numNei = GetCellEdgeNeighbors(
                    slots, slotKeys, slotCap, edgeCell, edgeNext, numPts, p1, p2, -1, cellIds
                )
                if numNei == 1:
                    # Compute new Position and PointData using the same
                    # subdivision scheme
                    nStencil[0] = 2
                    stencil[0] = Int64(p1)
                    stencil[1] = Int64(p2)
                    weights[0] = 0.5
                    weights[1] = 0.5
                elif numNei == 2:
                    GenerateOddStencil(
                        p1,
                        p2,
                        polysConn,
                        polysOff,
                        slots,
                        slotKeys,
                        slotCap,
                        edgeCell,
                        edgeNext,
                        numPts,
                        stencil,
                        weights,
                        cellIds,
                    )
                    nStencil[0] = 4
                else:
                    # Dataset is non-manifold and cannot be subdivided.
                    return 0
                newId = InterpolatePosition(
                    pts, stencil, Int(nStencil[0]), weights, outputPts, numOut, w
                )
            else:
                # we have already created a point on this edge. find it
                newId = FindEdge(
                    polysConn,
                    polysOff,
                    cellId,
                    p1,
                    p2,
                    edgeData,
                    slots,
                    slotKeys,
                    slotCap,
                    edgeCell,
                    edgeNext,
                    numPts,
                    neighbors,
                )
            edgeData[3 * cellId + edgeId] = Int64(newId)
            p1 = p2
            if edgeId < 2:
                p2 = Int(polysConn[Int(polysOff[cellId]) + edgeId + 1])
    counts[0] = Int64(numOut[0])
    counts[1] = Int64(
        GenerateSubdivisionCells(
            polysConn, polysOff, numCells, edgeData, oconn, ooffsets, outSrc
        )
    )
    return 1
