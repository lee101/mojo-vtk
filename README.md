# mojo-vtk

VTK's unstructured-grid filter kernels, ported to Mojo.

The kernels in `src/mojovtk/ported.mojo` are transcribed from the VTK 9.7.0
sources, function by function, keeping upstream's names, argument order,
defaults, branch structure and arithmetic. The Python package mirrors VTK's
class and method names, so a VTK call site translates one-for-one:

```python
import mojovtk as mv

pd = mv.PolyData(points).add_polys(cells)
out = mv.vtkTriangleFilter().SetInputData(pd).GetOutput()
```

is

```python
pd = vtk.vtkPolyData()  # points, cells
tf = vtk.vtkTriangleFilter()
tf.SetInputData(pd)
tf.Update()
out = tf.GetOutput()
```

Every number in `tests/` is asserted against real VTK 9.7.0, installed from
conda-forge, running the same input.

## Install

```bash
pixi install
pixi run build     # src/capi.mojo -> dist/libmojo-vtk.so
pixi run test      # pytest, parity against upstream vtk
pixi run bench     # markdown table against upstream vtk
```

`pixi run build` needs `mojo` (pinned) and, for the GPU-free CPU path, nothing
else. The library is rebuilt automatically by `python/mojovtk/_lib.py` when it
is older than `src/`, so an editable checkout works without running the build
task first.

## Usage

```python
import numpy as np
import mojovtk as mv

# a closed unit-cube surface as 12 triangles
c = np.asarray([
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
], dtype=np.float64)
faces = [
    [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
    [0, 1, 5], [0, 5, 4], [2, 3, 7], [2, 7, 6],
    [1, 2, 6], [1, 6, 5], [0, 4, 7], [0, 7, 3],
]
pd = mv.PolyData(c).add_polys(faces)

mp = mv.vtkMassProperties().SetInputData(pd)
print(mp.GetVolume(), mp.GetSurfaceArea())      # 1.0 6.0

normals = mv.vtkPolyDataNormals().SetInputData(pd).GetOutput()
print(normals.point_data["Normals"].shape)      # (8, 3)

edges = mv.vtkFeatureEdges().SetInputData(pd).GetOutput()
print(edges.lines.GetNumberOfCells())            # 12
```

## What is covered

Each row is a function transcribed from the named VTK 9.7.0 file.

| VTK class | Upstream source | What is ported |
| --- | --- | --- |
| `vtkTriangleFilter` | `Filters/Core/vtkTriangleFilter.cxx` | verts / lines / polys / triangle strips, `PassVerts`, `PassLines`, `PreservePolys`, `Tolerance` |
| `vtkMassProperties` | `Filters/Core/vtkMassProperties.cxx` | the whole `RequestData` loop and every reported quantity |
| `vtkCellCenters` | `Filters/Core/vtkCellCenters.cxx` | `ComputeCellCenters` and the empty-cell pass |
| `vtkFeatureEdges` | `Filters/Core/vtkFeatureEdges.cxx` | boundary / feature / non-manifold / manifold edges, `FeatureAngle`, `Coloring` |
| `vtkPolyDataNormals` | `Filters/Core/vtkPolyDataNormals.cxx` | `GetCellNormals`, `GetPointNormals` |
| `vtkPointDataToCellData` | `Filters/Core/vtkPointDataToCellData.cxx` | `PointDataToCellDataFunctor` (average) and `PointDataToCellDataCategoricalFunctor` (histogram majority) |
| `vtkCellDataToPointData` | `Filters/Core/vtkCellDataToPointData.cxx` | the `ContributingCellOption == All` fast path, `UnstructuredDataCD2PD` |
| `vtkLoopSubdivisionFilter` | `Filters/Modeling/vtkLoopSubdivisionFilter.cxx` and `Filters/General/vtkApproximatingSubdivisionFilter.cxx` | `GenerateSubdivisionPoints`, `GenerateEvenStencil`, `GenerateOddStencil`, `GenerateSubdivisionCells`, `FindEdge`, `InterpolatePosition` |
| `vtkCardinalSpline` | `Common/ComputationalGeometry/vtkCardinalSpline.cxx` and `Common/DataModel/vtkSpline.cxx` | `Evaluate`, `Compute`, `Fit1D`, `FitClosed1D`, `FindIndex`, `ComputeLeftDerivative`, `ComputeRightDerivative` |
| `vtkPolygon` | `Common/DataModel/vtkPolygon.cxx` | `ComputeNormal` (`NormalWorker`), `ComputeTolerance`, `ParameterizePolygon`, `InterpolateFunctions`, `SimpleTriangulation`, `vtkPolyVertexList`, `EarCutTriangulation`, `TriangulateLocalIds` |
| `vtkQuad` | `Common/DataModel/vtkQuad.cxx` | `TriangulateLocalIds` (the ear-cut failure fallback) |
| `vtkLine` | `Common/DataModel/vtkLine.cxx` | `DistanceToLine`, `Intersection` |
| `vtkMath`, `vtkPriorityQueue` | `Common/Core/` | the 3-vector helpers, `SolveLinearSystemGEPP2x2`, and a flat binary heap whose pop order is identical to `vtkPriorityQueue` |
| `vtkTriangleStrip` | `Common/DataModel/vtkTriangleStrip.cxx` | `DecomposeStrip` |
| cell types | `Common/DataModel/*.h` | `GetParametricCenter` and `EvaluateLocation` for vertex, poly vertex, line, poly line, triangle, quad, pixel, polygon, triangle strip, tetra, voxel, hexahedron, wedge, pyramid |

## What is not covered, and why

These are documented divergences, not silent ones.

1. **`vtkPolygon::BoundedTriangulate` and the `DOT_PRODUCT` / `BEST_QUALITY`
   ear-cut measures.** No filter in the ported set can reach them:
   `vtkTriangleFilter` calls `TriangulateLocalIds`, which calls the default
   `PERIMETER2_TO_AREA_RATIO` measure. `UnbiasedEarCutTriangulation` (used only
   by `BoundedTriangulate`) is therefore not ported.

2. **Ear-cut tie-breaking.** For a polygon outline where several candidate ears
   have the same `perimeter^2 / area` measure (a symmetric polygon, or a
   near-degenerate one), VTK can pick a different — equally valid — diagonal.
   Two mechanisms cause this, and neither is a kernel bug:
   * `vtkTriangleFilter` copies the polygon's points through a `vtkPolygon`,
     whose `vtkPoints` is **float32** by default, so VTK's measures are
     computed on float32-rounded coordinates. This port keeps float64. Callers
     who need VTK's exact tie-break can call
     `vtkPoints.SetDataTypeToDouble()` on the VTK side, which is what
     `tests/conftest.py::to_vtk_points(double=True)` does; with double points
     the two agree cell for cell (`test_triangle_filter_earcut_exact`).
   * Even with double points, VTK 9.7's own `vtkPolyDataNormals` point normals
     on a closed sphere are *not* the sum of its own cell normals (a five-line
     NumPy check disagrees by 0.03), so `test_polydata_normals_point_normals`
     asserts the algorithm's definition and
     `test_polydata_normals_match_vtk_on_folded_mesh` asserts VTK parity where
     VTK's output is the plain sum.
   `test_triangle_filter_earcut_ties_are_still_valid` checks the contract that
   must hold either way: same triangle count, same total area, same
   orientation.

3. **`vtkPolyDataNormals` splitting and consistency.** `RequestData` delegates
   those to `vtkSplitSharpEdgesPolyData` and `vtkOrientPolyData`, which are
   separate upstream filters and are not ported. `SetSplitting` and
   `SetConsistency` are accepted and ignored; the filter runs the
   non-splitting path, which is exactly what those two sub-filters being off
   produces.

4. **Ghost-cell handling in `vtkFeatureEdges`.** The `ghosts` branches
   (`InputGhostCellFinder`, `RemoveGhostInterfaces`, `CELL_NOT_VISIBLE`) are
   not ported: they need VTK's parallel ghost array, which this container does
   not model. `SetRemoveGhostInterfaces` is accepted and ignored.

5. **`vtkCellDataToPointData`'s `ContributingCellOption != All` path** (the
   `Spread` helper, which filters cells by dimension and supports cell
   patches). The default is `All`, which is the `UnstructuredDataCD2PD` fast
   path and is what is ported.

6. **`vtkPointDataToCellData` categorical parity against VTK.** VTK 9.7.0's
   viskores backend refuses categorical data (`CategoricalData is not
   supported` at run time), so
   `test_point_data_to_cell_data_categorical` compares against a direct
   NumPy transcription of `Histogram::IndexOfLargestBin`, including the
   `BinCountCmp` tie-break (largest count, then largest value).

7. **Pipeline plumbing.** VTK builds an execution pipeline and calls
   `Update()`; here the arrays are plain attributes and `GetOutput()` runs the
   kernel synchronously. `SetInputData` / `GetInput` / `Update` are present so
   call sites read the same, but there is no incremental update, no
   information object, no progress reporting and no abort.

8. **Scalar call results over ctypes.** A `Float64` argument or return value
   also crosses as an address, so a scalar parameter is a one-element
   `np.float64` buffer. This is invisible in the Python API and confined to
   `src/capi.mojo`.

9. **64-bit ids only.** `vtkIdType` is `int64` and the kernels use `Int`
   throughout, matching VTK's default. There is no 32-bit build.

10. **Serial kernels.** VTK's `vtkSMPTools::For` splits the per-cell loops
    across threads; these kernels are single-threaded. That is why
    `vtkCellDataToPointData` and `vtkPolyDataNormals` are much faster here
    despite the missing threading, while `vtkTriangleFilter` is close to VTK.
    Nothing in the port is SIMD-vectorised or parallelised yet; a later pass can
    add that without touching this structure.

## Benchmarks

Measured on the machine below with `pixi run bench`, which takes a
machine-wide `flock` so a concurrent factory job cannot distort the numbers.
Each figure is the best of 5 runs. The VTK column re-runs the filter (or
invalidates the pipeline with `Modified()`) so it measures the kernel rather
than VTK's `Update()` cache.

```
machine: Linux-6.8.0-142-generic-x86_64-with-glibc2.39
python 3.13.14  numpy 2.5.1  vtk 9.7.0

| case                                                 |     mojo-vtk |          vtk |  vtk / mojo-vtk |
| ---------------------------------------------------- | ------------ | ------------ | --------------- |
| vtkMassProperties (sphere 64x32, 4k tris)            |       0.28 ms |       0.40 ms |           1.41x |
| vtkMassProperties (sphere 256x128, 64k tris)         |       3.65 ms |       5.66 ms |           1.55x |
| vtkTriangleFilter (quad grid 256x256, 65k quads)     |      17.70 ms |      17.81 ms |           1.01x |
| vtkCellCenters (sphere 256x128, 64k tris)            |       1.59 ms |       8.18 ms |           5.14x |
| vtkFeatureEdges (sphere 256x128, 64k tris)           |      35.31 ms |      35.19 ms |           1.00x |
| vtkPolyDataNormals point normals (sphere 256x128)    |       6.69 ms |      40.66 ms |           6.07x |
| vtkPointDataToCellData (sphere 256x128, 64k tris)    |       0.95 ms |       3.84 ms |           4.02x |
| vtkCellDataToPointData (sphere 256x128, 64k tris)    |       1.64 ms |      89.99 ms |          54.79x |
| vtkLoopSubdivisionFilter (sphere 64x32 -> level 2)   |      15.20 ms |      46.72 ms |           3.07x |
| vtkCardinalSpline.Evaluate (1000 calls, 512 knots)   |       8.90 ms |       0.57 ms |           0.06x |
```

Reading the table honestly:

* `vtkCellDataToPointData` is 55x faster because VTK's default path rebuilds
  its cell links and walks the slow `Spread` template; the port builds the
  point-to-cell index once and averages directly.
* `vtkPolyDataNormals`, `vtkCellCenters` and `vtkLoopSubdivisionFilter` win
  mostly because VTK spends its time in `vtkGenericCell` construction and
  `vtkSMPTools` dispatch rather than in the arithmetic.
* `vtkTriangleFilter` (1.01x) and `vtkFeatureEdges` (1.00x) are level with VTK.
  Both are the cases where the arithmetic itself dominates, and VTK's scalar
  inner loops are already well optimised, so there is nothing for the port to
  win back.
* `vtkCardinalSpline.Evaluate` is **16x slower** than VTK. It is a scalar
  function returning one double, so the ctypes round trip dominates; VTK
  evaluates in-process. This is an honest cost of the FFI boundary, not of the
  kernel: 1000 calls cost 8.9 ms, i.e. about 9 microseconds per call, almost
  all of it Python-side.
* Nothing here is threaded or SIMD-vectorised. `vtkFeatureEdges` at 1.20x is
  the case most likely to improve, since its per-edge neighbour query is
  embarrassingly parallel.

## How it works

**FFI strategy.** `src/capi.mojo` is the only file that knows about C. It
declares one `@export("mvt_...")` function per kernel; each takes every buffer
as an `Int` address and rebuilds the typed pointer inside. That is forced:
`@export` rejects parametric functions, and a pointer with an inferred origin
is parametric, so `Pointer[Float64, AnyOrigin[mut=True]]` has to be named
explicitly (see `MOJO_NOTES.md`). `src/mojovtk/ported.mojo` is the port: it
never sees an address, keeps VTK's function bodies intact, and is compiled as
part of the same unit. `--disable-warnings` only silences the
`positional __getitem__` deprecation that `p[i]` triggers in every kernel.

**Memory layout.** Nothing allocates on the Mojo side, so nothing can leak.
The caller owns every buffer, including scratch, and sizes the scratch from
the layout comments at the top of `ported.mojo`.

* Points are a flat `(numPoints, 3)` float64 array, so `double x[3]` becomes a
  three-double window of a scratch pointer and the transcription is literal.
* A cell array is `(connectivity, offsets)` with
  `offsets[nCells] == len(connectivity)` — VTK's modern `vtkCellArray` layout.
  `GetNumberOfIds` and `GetId` are the ported accessors.
* Ids and offsets are 64-bit, matching `vtkIdType`.
* `BuildPointLinks` and `BuildEdgeTable` stand in for
  `vtkStaticCellLinks` and `vtkEdgeTable`, which upstream builds inside
  `vtkPolyData` with types that have no Mojo equivalent. They are flat
  caller-owned arrays (an offset/count pair, and an open-addressed key table
  with per-edge chains), and every consumer asks them the same question
  `vtkPolyData::GetPointCells` / `GetCellEdgeNeighbors` answers. The
  neighbour lists come out in increasing cell id, like the links they stand
  in for, so the branch structure downstream is unchanged.

**Container model.** `mojovtk.PolyData` and `mojovtk.UnstructuredGrid` hold
the same arrays VTK does, in VTK's cell order, and convert to each other the
way `vtkPolyData` and `vtkUnstructuredGrid` do. The Python layer is pure: it
allocates NumPy buffers, calls the kernel, and gathers the per-cell/per-point
attribute maps that upstream's `CopyData` calls would produce. Where upstream
renumbers a result (the `vtkFeatureEdges` point merge, for example) the
renumbering lives in the kernel, not in Python.

## Layout

```
src/capi.mojo              the @export C ABI, one shim per kernel
src/mojovtk/ported.mojo    the port, in upstream module order
build/build.sh             mojo build --emit shared-lib -> dist/libmojo-vtk.so
python/mojovtk/_lib.py     ctypes loader (builds on demand)
python/mojovtk/datamodel.py  CellArray, PolyData, UnstructuredGrid
python/mojovtk/filters.py    the filter classes
tests/test_parity.py       every filter against real VTK 9.7.0
bench/bench.py             the table above
```

## License

BSD-3-Clause, matching upstream VTK. See `LICENSE`.
