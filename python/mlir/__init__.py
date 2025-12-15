"""
Compatibility shim for code that expects the upstream `mlir` Python package.

RocDSL embeds MLIR's Python bindings under the `_mlir` package (via MLIR's
official AddMLIRPython integration). This `mlir` package provides lightweight
proxy modules that forward to `_mlir` so older code can keep doing:

  - `from mlir import ir`
  - `from mlir.dialects import func, arith, ...`

Critically, these proxies avoid importing MLIR's extension libraries under two
different module name prefixes (which would crash due to duplicate global
initialization).
"""

from __future__ import annotations

__all__ = ["ir", "passmanager", "rewrite", "execution_engine", "dialects"]


