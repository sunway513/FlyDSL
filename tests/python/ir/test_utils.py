"""Utility functions for IR tests."""


def unwrap_values(*values):
    """Unwrap ArithValue objects to get underlying MLIR Values.
    
    This is needed because ArithValue wraps MLIR Values for operator
    overloading, but func.FuncOp.from_py_func expects raw MLIR Values.
    
    Args:
        *values: One or more values (may be ArithValue or MLIR Value)
    
    Returns:
        Tuple of unwrapped MLIR Values
    
    Example:
        >>> @func.FuncOp.from_py_func()
        >>> def my_func():
        >>>     result1 = some_operation()
        >>>     result2 = another_operation()
        >>>     return unwrap_values(result1, result2)
    """
    unwrapped = []
    for val in values:
        if hasattr(val, '_value'):
            # It's an ArithValue or similar wrapper
            unwrapped.append(val._value)
        else:
            # Already an MLIR Value
            unwrapped.append(val)
    return tuple(unwrapped)

