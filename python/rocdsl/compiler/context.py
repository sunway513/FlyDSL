import contextlib
import importlib
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Import from embedded MLIR Python modules
try:
    # First try to import from the build directory (development mode)
    _ROCIR_PYTHON_PATH = Path(__file__).resolve().parents[3] / "build" / "python_packages" / "rocdsl"
    if _ROCIR_PYTHON_PATH.exists():
        _mlir_path = str(_ROCIR_PYTHON_PATH / "_mlir")
        if _mlir_path not in sys.path:
            sys.path.insert(0, _mlir_path)
    
    from _mlir import ir
except ImportError:
    # Fallback to system-installed mlir (for backward compatibility)
    from mlir import ir

_PASSES_MODULE = None


def ensure_rocir_python_extensions(context: ir.Context):
    """Ensure Rocir passes and dialect are registered for the given context."""
    global _PASSES_MODULE
    if _PASSES_MODULE is None:
        # Import the embedded passes module
        try:
            from _mlir._mlir_libs import _rocirPasses as _PASSES_MODULE
        except ImportError:
            # Fallback for old build system
            import _rocirPasses as _PASSES_MODULE
    
    # Register dialects using the new nanobind interface
    from _mlir import ir as mlir_ir
    dialect_registry = mlir_ir.DialectRegistry()
    _PASSES_MODULE.register_dialects(dialect_registry._CAPIPtr)
    context.append_dialect_registry(dialect_registry)


@dataclass
class MLIRContext:
    context: ir.Context
    module: ir.Module

    def __str__(self):
        return str(self.module)


class RAIIMLIRContext:
    context: ir.Context
    location: ir.Location

    def __init__(self, location: Optional[ir.Location] = None, allow_unregistered_dialects=False):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        ensure_rocir_python_extensions(self.context)
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()

    def __del__(self):
        self.location.__exit__(None, None, None)
        self.context.__exit__(None, None, None)
        if ir is not None:
            assert ir.Context is not self.context


class RAIIMLIRContextModule:
    context: ir.Context
    location: ir.Location
    insertion_point: ir.InsertionPoint
    module: ir.Module

    def __init__(self, location: Optional[ir.Location] = None, allow_unregistered_dialects=False):
        self.context = ir.Context()
        if allow_unregistered_dialects:
            self.context.allow_unregistered_dialects = True
        self.context.__enter__()
        ensure_rocir_python_extensions(self.context)
        if location is None:
            location = ir.Location.unknown()
        self.location = location
        self.location.__enter__()
        self.module = ir.Module.create()
        self.insertion_point = ir.InsertionPoint(self.module.body)
        self.insertion_point.__enter__()

    def __del__(self):
        self.insertion_point.__exit__(None, None, None)
        self.location.__exit__(None, None, None)
        self.context.__exit__(None, None, None)
        if ir is not None:
            assert ir.Context is not self.context


class ExplicitlyManagedModule:
    module: ir.Module
    _ip: ir.InsertionPoint

    def __init__(self):
        self.module = ir.Module.create()
        self._ip = ir.InsertionPoint(self.module.body)
        self._ip.__enter__()

    def finish(self):
        self._ip.__exit__(None, None, None)
        return self.module

    def __str__(self):
        return str(self.module)


@contextlib.contextmanager
def enable_multithreading(context=None):
    if context is None:
        context = ir.Context.current
    context.enable_multithreading(True)
    yield
    context.enable_multithreading(False)


@contextlib.contextmanager
def disable_multithreading(context=None):
    if context is None:
        context = ir.Context.current

    context.enable_multithreading(False)
    yield
    context.enable_multithreading(True)


@contextlib.contextmanager
def enable_debug():
    ir._GlobalDebug.flag = True
    yield
    ir._GlobalDebug.flag = False
