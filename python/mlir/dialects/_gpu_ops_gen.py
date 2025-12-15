"""Proxy module: `mlir.dialects._gpu_ops_gen` -> `_mlir.dialects._gpu_ops_gen`.

Note: we must re-export `_Dialect`, which may not be part of `__all__`.
"""

from __future__ import annotations

from _mlir.dialects import _gpu_ops_gen as _src  # type: ignore

# Re-export everything (including private helpers) so downstream imports like
# `from mlir.dialects._gpu_ops_gen import _Dialect` keep working.
for _k, _v in _src.__dict__.items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v

# Export everything public by default so downstream `import *` gets the op
# classes (e.g. GPUModuleOp) even if upstream `__all__` is conservative.
__all__ = [k for k in globals().keys() if not k.startswith("_")]


