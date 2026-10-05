#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# `MODULAR_HOME` must point at the toolchain's `share/max` or every import,
# `std` included, fails to resolve. `pixi run` exports it; a shell that only
# has `mojo` on PATH does not, so derive it from `<prefix>/bin/mojo`.
if [[ -z "${MODULAR_HOME:-}" && -x "$(command -v mojo || true)" ]]; then
    prefix="$(dirname "$(dirname "$(readlink -f "$(command -v mojo)")")")"
    [[ -d "$prefix/share/max" ]] && export MODULAR_HOME="$prefix/share/max"
fi

# `--disable-warnings` only silences the `positional __getitem__` deprecation,
# which every `p[i]` in the kernels trips; the build is warning-clean otherwise.
mojo build --emit shared-lib --disable-warnings -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-vtk.so"

# `python/mojovtk/_lib.py` loads `dist/libmojo-vtk.so` and rebuilds it when it
# is older than `src/`, so keep that copy fresh instead of paying a second build.
cp "$repo_dir/dist/libmojo-vtk.so" "$repo_dir/build/capi.so"
