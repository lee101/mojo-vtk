"""mojo-vtk: VTK's unstructured-grid filter kernels, ported to Mojo.

The API mirrors upstream names and defaults, so a VTK call site
(`vtkTriangleFilter()` -> `SetInputData(...)` -> `Update()`) translates
one-for-one to `mojovtk.vtkTriangleFilter()` -> `SetInputData(...)` ->
`GetOutput()`.  See the README for what is covered and what is not.
"""

from .datamodel import (  # noqa: F401
    CellArray,
    GetClassNameFromTypeId,
    GetDimension,
    PolyData,
    UnstructuredGrid,
    VTK_EMPTY_CELL,
    VTK_HEXAHEDRON,
    VTK_HEXAGONAL_PRISM,
    VTK_INVALID,
    VTK_LINE,
    VTK_PENTAGONAL_PRISM,
    VTK_PIXEL,
    VTK_POLYGON,
    VTK_POLY_LINE,
    VTK_POLY_VERTEX,
    VTK_PYRAMID,
    VTK_QUAD,
    VTK_TETRA,
    VTK_TRIANGLE,
    VTK_TRIANGLE_STRIP,
    VTK_VERTEX,
    VTK_VOXEL,
    VTK_WEDGE,
)
from .filters import (  # noqa: F401
    vtkCardinalSpline,
    vtkCellCenters,
    vtkCellDataToPointData,
    vtkFeatureEdges,
    vtkLoopSubdivisionFilter,
    vtkMassProperties,
    vtkPointDataToCellData,
    vtkPolyDataNormals,
    vtkTriangleFilter,
)

__version__ = "0.1.0"
