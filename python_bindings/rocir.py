"""Rocir Dialect for MLIR Python Bindings"""

from mlir import ir
from mlir.dialects._ods_common import _cext as _ods_cext

# Import generated ops
from _rocir_ops_gen import *
from _rocir_ops_gen import _Dialect

# Register the dialect when imported
try:
    # The dialect is automatically registered via @_ods_cext.register_dialect decorator
    # in _rocir_ops_gen.py, but we need to ensure it's loaded in the context
    pass
except Exception as e:
    import warnings
    warnings.warn(f"Failed to register Rocir dialect: {e}")

__all__ = []
