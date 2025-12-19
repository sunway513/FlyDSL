from __future__ import annotations

import inspect
from typing import Optional

from _mlir import ir
from _mlir.dialects import func as mlir_func

from rocdsl.compiler.context import ensure_rocir_python_extensions
from rocdsl.dialects.ext import gpu
from rocdsl.dialects.ext.func import prep_func_types


class MlirModule:
    GPU_MODULE_NAME = "kernels"

    cls_kernel_fn = []
    cls_jit_fn = []
    cls_kernel_sym = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Each MlirModule subclass owns a dedicated MLIR context.
        # We intentionally avoid process-global `Context.__enter__()` at import time:
        # some environments crash during interpreter shutdown if a context is left
        # on the thread-local stack.
        cls._context = ir.Context()
        ensure_rocir_python_extensions(cls._context)
        cls._location = ir.Location.unknown(cls._context)

        with cls._context, cls._location:
            # Initialize MLIR module for this subclass.
            cls.module = ir.Module.create()
            gpu.set_container_module(cls.module)

            # Create gpu.module container.
            with ir.InsertionPoint(cls.module.body):
                cls.gpu_module = gpu.GPUModuleOp(cls.GPU_MODULE_NAME)

        # Collect descriptor wrappers for this class (not inherited).
        temp_kernel_fn = []
        temp_jit_fn = []
        temp_kernel_sym = {}
        for name, value in cls.__dict__.items():
            if isinstance(value, _KernelDescriptor) and hasattr(value, "_wrapper"):
                temp_kernel_fn.append(value._wrapper)
                temp_kernel_sym[name] = name
            elif isinstance(value, _JitDescriptor) and hasattr(value, "_wrapper"):
                temp_jit_fn.append(value._wrapper)

        cls.cls_kernel_fn = temp_kernel_fn
        cls.cls_jit_fn = temp_jit_fn
        cls.cls_kernel_sym = temp_kernel_sym

    def __init__(self):
        self.kernel_func_op = {}
        with self._context, self._location:
            # Emit host-side jit functions first.
            for fn in self.cls_jit_fn:
                fn(self)
            # Emit GPU kernels.
            for fn in self.cls_kernel_fn:
                fn(self)

    def __repr__(self):
        return str(self.module)

    def __getattr__(self, name: str):
        # Match Flyx: resolve kernel symbol reference via attribute.
        if name in self.cls_kernel_sym:
            with self._context, self._location:
                return ir.SymbolRefAttr.get([self.GPU_MODULE_NAME, self.cls_kernel_sym[name]])
        raise AttributeError(f"{name} not found in kernel functions.")

    @classmethod
    def kernel(cls, fn):
        """Alternative classmethod decorator (non-descriptor), for completeness."""

        def wrapper(self, *args, **kwargs):
            with self._context, self._location:
                with ir.InsertionPoint.at_block_begin(self.gpu_module.body):
                    # Emit as gpu.func in the gpu.module region.
                    k = gpu.func(fn, emit=True, lower_range_loops=True)
                    # Ensure launch path qualifies as `@kernels::fn` when used as GPUFunc callable.
                    try:
                        k.qualname = self.GPU_MODULE_NAME
                    except Exception:
                        pass
                    self.kernel_func_op[fn.__name__] = k

        cls.cls_kernel_fn.append(wrapper)
        cls.cls_kernel_sym[fn.__name__] = fn.__name__
        return fn

    @classmethod
    def jit(cls, fn):
        """Alternative classmethod decorator (non-descriptor), for completeness."""

        def wrapper(self):
            with self._context, self._location:
                with ir.InsertionPoint.at_block_begin(self.module.body):
                    sig = inspect.signature(fn)
                    input_types, _, _ = prep_func_types(sig, [])
                    mlir_func.FuncOp.from_py_func(*input_types)(fn)

        cls.cls_jit_fn.append(wrapper)
        return fn


class _KernelDescriptor:
    """Descriptor that registers kernel to the owning MlirModule subclass."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        try:
            if issubclass(owner, MlirModule):
                fn = self.fn

                def wrapper(instance_self, *args, **kwargs):
                    with instance_self._context, instance_self._location:
                        with ir.InsertionPoint.at_block_begin(instance_self.gpu_module.body):
                            k = gpu.func(fn, emit=True, lower_range_loops=True)
                            try:
                                k.qualname = instance_self.GPU_MODULE_NAME
                            except Exception:
                                pass
                            instance_self.kernel_func_op[fn.__name__] = k

                self._wrapper = wrapper
                self._name = name
        except TypeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


class _JitDescriptor:
    """Descriptor that registers a host-side jit function to MlirModule."""

    def __init__(self, fn):
        self.fn = fn
        self.name = fn.__name__
        self._wrapper = None

    def __set_name__(self, owner, name):
        try:
            if issubclass(owner, MlirModule):
                fn = self.fn

                def wrapper(instance_self):
                    with instance_self._context, instance_self._location:
                        with ir.InsertionPoint.at_block_begin(instance_self.module.body):
                            sig = inspect.signature(fn)
                            input_types, _, _ = prep_func_types(sig, [])
                            mlir_func.FuncOp.from_py_func(*input_types)(fn)

                self._wrapper = wrapper
        except TypeError:
            pass

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.fn
        return self.fn.__get__(obj, objtype)


def kernel(fn):
    """Descriptor decorator: use as `@roc.lang.kernel` on methods."""
    return _KernelDescriptor(fn)


def jit(fn):
    """Descriptor decorator: use as `@roc.lang.jit` on methods."""
    return _JitDescriptor(fn)




