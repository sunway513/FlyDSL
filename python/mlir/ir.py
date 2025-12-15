"""Proxy module: `mlir.ir` -> `_mlir.ir`.

This proxy also makes `Context()` more user-friendly for this repo's tests by
auto-loading all available dialects when entering the context manager.
"""

from __future__ import annotations

from _mlir.ir import *  # type: ignore

from _mlir import ir as _ir  # type: ignore


class Context(_ir.Context):  # type: ignore[misc]
    """Context that auto-loads all available dialects on `with ctx:`."""

    def __enter__(self):
        # Register and load all upstream dialects, plus Rocir, so parsing
        # succeeds for common ops like `func.func` in tests.
        try:
            from _mlir._mlir_libs import _rocirPasses  # type: ignore

            reg = _ir.DialectRegistry()
            _rocirPasses.register_dialects(reg._CAPIPtr)
            self.append_dialect_registry(reg)
        except Exception:
            # If the extension isn't available, we'll fall back to whatever is
            # already registered.
            pass

        try:
            self.load_all_available_dialects()
        except Exception:
            pass
        return super().__enter__()


