"""RocDSL dialects"""

import sys
import os
from pathlib import Path

# Add build directory to Python path for embedded MLIR modules
_rocdsl_root = Path(__file__).resolve().parents[3]
_python_packages_dir = _rocdsl_root / "build" / "python_packages" / "rocdsl"
if _python_packages_dir.exists():
    _python_packages_str = str(_python_packages_dir)
    if _python_packages_str not in sys.path:
        sys.path.insert(0, _python_packages_str)

# Import the rocir dialect from embedded modules
try:
    from _mlir.dialects import rocir
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import rocir dialect: {e}")
