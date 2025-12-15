"""RocDSL - ROCm Domain Specific Language for layout algebra"""

__version__ = "0.1.0"

# Setup Python path for embedded MLIR modules
import sys
import os
from pathlib import Path

# Add the build directory to Python path (development mode)
_rocdsl_root = Path(__file__).resolve().parents[2]
_python_packages_dir = _rocdsl_root / "build" / "python_packages" / "rocdsl"
if _python_packages_dir.exists():
    _python_packages_str = str(_python_packages_dir)
    if _python_packages_str not in sys.path:
        sys.path.insert(0, _python_packages_str)

# Lazy import dialects and passes to avoid requiring MLIR when only using runtime
def __getattr__(name):
    if name == "rocir":
        from .dialects.ext import rocir
        return rocir
    elif name == "arith":
        from .dialects.ext import arith
        return arith
    elif name == "scf":
        from .dialects.ext import scf
        return scf
    elif name in ["Pipeline", "run_pipeline", "lower_rocir_to_standard"]:
        from . import passes
        return getattr(passes, name)

__all__ = [
    "rocir",
    "arith",
    "scf",
    "Pipeline",
    "run_pipeline",
    "lower_rocir_to_standard",
]

# Export compiler modules
from .compiler import Pipeline, run_pipeline
from .compiler.context import RAIIMLIRContextModule

__all__.extend(["Pipeline", "run_pipeline", "RAIIMLIRContextModule"])
