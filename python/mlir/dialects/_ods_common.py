"""Proxy module: `mlir.dialects._ods_common` -> `_mlir.dialects._ods_common`.

Downstream code may import private helpers like `_cext`, so we must re-export
more than just `__all__`.
"""

from __future__ import annotations

from _mlir.dialects import _ods_common as _src  # type: ignore

for _k, _v in _src.__dict__.items():
    if _k.startswith("__"):
        continue
    globals()[_k] = _v

__all__ = getattr(_src, "__all__", [])  # type: ignore


