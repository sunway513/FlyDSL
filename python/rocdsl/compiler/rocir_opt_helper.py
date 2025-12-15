"""Helper utilities to run Rocir coordinate lowering via the MLIR Python API.

This module is maintained for backward compatibility. New code should use
the Pipeline class from rocdsl.compiler.pipeline instead.
"""

from _mlir.ir import Module as ir_Module

# Import from the new pipeline module
from rocdsl.compiler.pipeline import Pipeline, RocDSLCompilerError

# Legacy constant for backward compatibility
_ROCIR_COORD_LOWERING_PIPELINE = "builtin.module(rocir-coord-lowering)"


def apply_rocir_coord_lowering(module: ir_Module) -> ir_Module:
    """Apply the rocir-coord-lowering pass using the in-process PassManager.

    The pass mutates the provided module in-place and also returns it for
    convenience so callers can chain additional processing.

    Note: This function is maintained for backward compatibility. New code
    should use Pipeline().rocir_coord_lowering().run(module) instead.

    Args:
        module: MLIR module containing Rocir coordinate operations.

    Returns:
        The same module instance with Rocir ops lowered to arithmetic.

    Raises:
        RuntimeError: If the pass pipeline fails.
    
    Example (new API):
        >>> from rocdsl.compiler.pipeline import Pipeline
        >>> pipeline = Pipeline().rocir_coord_lowering()
        >>> result = pipeline.run(module)
    """
    try:
        pipeline = Pipeline().rocir_coord_lowering()
        return pipeline.run(module)
    except RocDSLCompilerError as exc:
        # Re-raise as RuntimeError for backward compatibility
        raise RuntimeError(str(exc)) from exc
