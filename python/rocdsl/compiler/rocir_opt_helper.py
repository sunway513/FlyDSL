"""Helper to apply rocir-opt passes via subprocess"""

import subprocess
import tempfile
import os
from mlir import ir


def apply_rocir_coord_lowering(module: ir.Module) -> ir.Module:
    """Apply rocir-coord-lowering pass using rocir-opt subprocess
    
    Args:
        module: MLIR module containing rocir coordinate operations
        
    Returns:
        New MLIR module with rocir ops lowered to arithmetic
        
    Raises:
        FileNotFoundError: If rocir-opt binary not found
        subprocess.CalledProcessError: If rocir-opt execution fails
    """
    # Import rocir to ensure dialect is registered globally
    import sys
    sys.path.insert(0, '/mnt/raid0/felix/rocDSL/build/python_bindings')
    try:
        import rocir_ops  # This registers the rocir dialect
    except ImportError:
        pass  # May already be imported
    
    # Convert module to MLIR text
    module_str = str(module)
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.mlir', delete=False) as f:
        f.write(module_str)
        temp_input = f.name
    
    try:
        # Find rocir-opt binary
        rocir_opt = '/mnt/raid0/felix/rocDSL/build/tools/rocir-opt/rocir-opt'
        if not os.path.exists(rocir_opt):
            import shutil
            rocir_opt = shutil.which('rocir-opt')
            if not rocir_opt:
                raise FileNotFoundError("rocir-opt binary not found in build/tools/rocir-opt or PATH")
        
        # Run rocir-opt with the lowering pass
        result = subprocess.run(
            [rocir_opt, temp_input, '--allow-unregistered-dialect', '--rocir-coord-lowering'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            print(f"rocir-opt stderr:\n{result.stderr}")
            print(f"rocir-opt stdout:\n{result.stdout}")
            raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        
        # Parse with new context - rocir should now be registered
        new_ctx = ir.Context()
        new_ctx.allow_unregistered_dialects = True
        return ir.Module.parse(result.stdout, new_ctx)
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_input):
            os.unlink(temp_input)
