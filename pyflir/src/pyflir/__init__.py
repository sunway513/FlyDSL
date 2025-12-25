"""FLIR - ROCm Domain Specific Language for layout algebra"""

__version__ = "0.1.0"

# Setup Python path for embedded MLIR modules
import sys
import os
import importlib
from pathlib import Path
from pkgutil import extend_path

# Development convenience:
# In this repo we often have *two* copies of the `pyflir` package:
# - source tree:      `pyflir/src/pyflir`
# - build output:     `.flir/build/python_packages/pyflir/pyflir`
#
# Most test runs end up importing the build-output package because
# `.flir/build/python_packages/pyflir` is on `sys.path`. That makes edits to the
# source tree appear to "not work".
#
# To make `pyflir/src/...` edits take effect immediately (without rebuilding),
# we treat `pyflir` as a multi-path package and *prefer* the source tree for
# submodules when it exists.
__path__ = extend_path(__path__, __name__)  # type: ignore[name-defined]

_this = Path(__file__).resolve()
_repo_root = None
for _p in _this.parents:
    _src_pkg = _p / "pyflir" / "src" / "pyflir"
    if _src_pkg.is_dir():
        _repo_root = _p
        _src_pkg_str = str(_src_pkg)
        if _src_pkg_str not in __path__:  # type: ignore[operator]
            __path__.insert(0, _src_pkg_str)  # type: ignore[operator]
        break
if _repo_root is None:
    # Fallback: best-effort guess (keeps installed/wheel layouts working).
    _repo_root = _this.parents[2]

# IMPORTANT:
# Do not blindly prepend build/python_packages/pyflir to sys.path.
# That directory contains an embedded `_mlir` package which can conflict with an
# external MLIR python runtime (mlir_core), leading to crashes like:
#   LLVM ERROR: Option 'basic' already exists!
#
# If you explicitly want to use the embedded MLIR runtime, set:
#   FLIR_USE_EMBEDDED_MLIR=1
# For backward compatibility (pre-rename), we also honor:
#   FLIR_USE_EMBEDDED_MLIR=1
_flir_root = _repo_root
_use_embedded = os.environ.get("FLIR_USE_EMBEDDED_MLIR")
if _use_embedded is None:
    _use_embedded = os.environ.get("FLIR_USE_EMBEDDED_MLIR", "0")

def _resolve_embedded_python_packages_dir() -> Path | None:
    """Return the repo build's python_packages/<project> directory, if it exists."""
    # Default build layout: `.flir/build` (see build.sh/setup.py), with fallback to legacy `build/`.
    _build_dir = os.environ.get("FLIR_BUILD_DIR") or os.environ.get("FLIR_BUILD_DIR")
    if _build_dir is None:
        _build_dir_path = _flir_root / ".flir" / "build"
        if not _build_dir_path.exists():
            _build_dir_path = _flir_root / "build"
    else:
        _build_dir_path = Path(_build_dir)
        if not _build_dir_path.is_absolute():
            _build_dir_path = _flir_root / _build_dir_path

    # New layout: python_packages/pyflir (and legacy python_packages/flir, python_packages/rocdsl).
    for _name in ("pyflir", "flir", "rocdsl"):
        _p = _build_dir_path / "python_packages" / _name
        if _p.exists():
            return _p
    return None


def _ensure_mlir_runtime_on_path(force: bool) -> None:
    """Make `_mlir` importable by prepending the embedded runtime path when needed.

    We avoid *always* prepending the embedded runtime to prevent conflicts with
    an external MLIR python runtime. However, in many dev/test environments the
    external runtime is absent; in that case, importing `pyflir` would fail
    early (`from _mlir import ir`). This fallback keeps direct script runs working.
    """
    # If a `_mlir` runtime is already discoverable, do nothing (prevents conflicts).
    if not force:
        try:
            if importlib.util.find_spec("_mlir") is not None:
                return
        except Exception:
            # Best-effort; proceed to attempt embedded fallback.
            pass

    _python_packages_dir = _resolve_embedded_python_packages_dir()
    if _python_packages_dir is None:
        return
    _python_packages_str = str(_python_packages_dir)
    if _python_packages_str not in sys.path:
        sys.path.insert(0, _python_packages_str)


# If the user explicitly requests the embedded MLIR runtime, always add it.
# Otherwise, only add it when `_mlir` isn't importable (common with PEP660 editable installs).
_ensure_mlir_runtime_on_path(force=(_use_embedded == "1"))

# Lazy import dialects and passes to avoid requiring MLIR when only using runtime
def __getattr__(name):
    if name == "flir":
        return importlib.import_module(".dialects.ext.flir", __name__)
    elif name == "arith":
        return importlib.import_module(".dialects.ext.arith", __name__)
    elif name == "scf":
        return importlib.import_module(".dialects.ext.scf", __name__)
    elif name == "lang":
        return importlib.import_module(".lang", __name__)
    elif name in ["Pipeline", "run_pipeline", "lower_flir_to_standard"]:
        from . import passes
        return getattr(passes, name)
    elif name == "compile":
        from .compiler.compiler import compile
        return compile
    elif name == "Executor":
        from .compiler.executor import Executor
        return Executor

__all__ = [
    "flir",
    "arith",
    "scf",
    "lang",
    "Pipeline",
    "run_pipeline",
    "lower_flir_to_standard",
    "compile",
    "Executor",
]

# Export compiler modules (safe imports only).
from .compiler import Pipeline, run_pipeline
from .compiler.context import RAIIMLIRContextModule

__all__.extend(["Pipeline", "run_pipeline", "RAIIMLIRContextModule", "compile", "Executor"])
