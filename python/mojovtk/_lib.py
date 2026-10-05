"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "src")
LIB = os.environ.get("MOJOVTK_LIB") or os.path.join(ROOT, "dist", "libmojo-vtk.so")

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype).  Every buffer crosses as an address.
_SIGNATURES = {
    "mvt_triangle_filter": ([I] * 28, None),
    "mvt_cell_centers": ([I] * 10, None),
    "mvt_mass_properties": ([I] * 6, I),
    "mvt_feature_edges": ([I] * 20, None),
    "mvt_polydata_normals_cell_normals": ([I] * 9, None),
    "mvt_polydata_normals_point_normals": ([I] * 11, None),
    "mvt_point_data_to_cell_data": ([I] * 7, None),
    "mvt_point_data_to_cell_data_categorical": ([I] * 7, I),
    "mvt_cell_data_to_point_data": ([I] * 10, None),
    "mvt_polygon_compute_normal": ([I] * 6, I),
    "mvt_polygon_triangulate": ([I] * 13, I),
    "mvt_decompose_strip": ([I] * 6, I),
    "mvt_build_point_links": ([I] * 7, I),
    "mvt_cardinal_spline_compute": ([I] * 13, I),
    "mvt_cardinal_spline_evaluate": ([I] * 4, F),
    "mvt_loop_subdivision": ([I] * 13, I),
}


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOVTK_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOVTK_MOJO=/path/to/mojo")


def subprocess_env(cmd: list[str]) -> dict:
    """Environment for `mojo build`.

    Every `import`, including `std`, resolves through `MODULAR_HOME`; without
    it the compiler reports `unable to locate module 'std'`, which looks like a
    missing package. `pixi run` exports it, a bare `mojo build` does not, so
    derive it from the toolchain layout `<prefix>/bin/mojo`, `share/max`.
    """
    env = os.environ.copy()
    if env.get("MODULAR_HOME"):
        return env
    exe = shutil.which(cmd[0]) or cmd[0]
    if os.path.basename(exe) == "mojo":
        share = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(exe))), "share", "max")
        if os.path.isdir(share):
            env["MODULAR_HOME"] = share
    return env


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-vtk.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB), exist_ok=True)
    cmd = mojo_command() + [
        "build", "--emit", "shared-lib", "--disable-warnings", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, env=subprocess_env(cmd), capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.array(a, dtype=np.float64, order="C", copy=True) if copy else \
        np.ascontiguousarray(a, dtype=np.float64)
    return arr


def i64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous int64 view of `a`; ids are 64-bit, as vtkIdType is."""
    arr = np.array(a, dtype=np.int64, order="C", copy=True) if copy else \
        np.ascontiguousarray(a, dtype=np.int64)
    return arr


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def main() -> int:
    """`python -m mojovtk._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
