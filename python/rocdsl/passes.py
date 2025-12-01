"""Pass management for RocDSL compiler.

This module provides the main interface for defining and running
transformation passes on MLIR modules.
"""

# Re-export everything from compiler.pipeline
from .compiler.pipeline import (
    Pipeline,
    RocDSLCompilerError,
    run_pipeline,
    lower_rocir_to_standard,
    apply_rocir_coord_lowering,
)

__all__ = [
    "Pipeline",
    "RocDSLCompilerError",
    "run_pipeline",
    "lower_rocir_to_standard",
    "apply_rocir_coord_lowering",
]
