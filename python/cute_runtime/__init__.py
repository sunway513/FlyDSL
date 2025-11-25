"""
CuTe Runtime Python API
========================

High-level Python interface for compiling and executing CuTe kernels.

Example usage:
    >>> import rocir_runtime as cute
    >>> import numpy as np
    
    >>> # Create GEMM executor
    >>> gemm = cute.Gemm(M=1024, N=1024, K=1024, arch='sm90')
    
    >>> # Compile from MLIR
    >>> mlir_code = '''
    ... func.func @cute_gemm(%A: memref<1024x1024xf16>, 
    ...                       %B: memref<1024x1024xf16>,
    ...                       %C: memref<1024x1024xf32>) {
    ...   // CuTe IR kernel implementation
    ... }
    ... '''
    >>> gemm.compile(mlir_code)
    
    >>> # Execute
    >>> A = np.random.randn(1024, 1024).astype(np.float16)
    >>> B = np.random.randn(1024, 1024).astype(np.float16)
    >>> C = gemm(A, B)
"""

from typing import Optional, Tuple, Union
import numpy as np
from pathlib import Path

# Import C++ bindings
try:
    from . import _cute_bindings as _cpp
except ImportError:
    raise ImportError(
        "CuTe runtime C++ bindings not found. "
        "Please build the extension with: python setup.py install"
    )

__version__ = "0.1.0"
__all__ = [
    "Gemm", "Kernel", "Compiler", 
    "DeviceBuffer", "TMADescriptor",
    "compile_mlir", "get_device_info"
]

#===----------------------------------------------------------------------===#
# Device Info
#===----------------------------------------------------------------------===#

def get_device_info(device_id: int = 0) -> dict:
    """
    Get CUDA device properties.
    
    Args:
        device_id: CUDA device ID (default: 0)
    
    Returns:
        Dictionary with device properties:
        - name: Device name
        - compute_capability: (major, minor) tuple
        - total_memory: Total global memory in bytes
        - multiprocessor_count: Number of SMs
    """
    props = _cpp.get_device_properties(device_id)
    return {
        'name': props.name,
        'compute_capability': (props.major, props.minor),
        'total_memory': props.total_global_mem,
        'multiprocessor_count': props.multi_processor_count,
        'max_threads_per_block': props.max_threads_per_block,
        'max_shared_memory_per_block': props.shared_mem_per_block
    }

#===----------------------------------------------------------------------===#
# Compiler Interface
#===----------------------------------------------------------------------===#

class Compiler:
    """MLIR → PTX/CUBIN compiler."""
    
    def __init__(self, mlir_bin_path: Optional[str] = None):
        """
        Initialize compiler.
        
        Args:
            mlir_bin_path: Path to MLIR tools (mlir-opt, mlir-translate)
        """
        self._compiler = _cpp.Compiler()
        if mlir_bin_path:
            self._compiler.set_mlir_bin_path(mlir_bin_path)
    
    def compile_to_ptx(
        self, 
        mlir_code: str, 
        arch: str = 'sm90',
        opt_level: int = 2
    ) -> str:
        """
        Compile MLIR to PTX.
        
        Args:
            mlir_code: MLIR source code
            arch: Target architecture (sm80, sm90, sm100)
            opt_level: Optimization level (0-3)
        
        Returns:
            PTX assembly code
        """
        arch_enum = self._parse_arch(arch)
        return self._compiler.compile_to_ptx(mlir_code, arch_enum, opt_level)
    
    def compile_to_cubin(
        self,
        ptx_code: str,
        arch: str = 'sm90'
    ) -> str:
        """
        Compile PTX to CUBIN.
        
        Args:
            ptx_code: PTX assembly code
            arch: Target architecture
        
        Returns:
            Path to compiled CUBIN file
        """
        arch_enum = self._parse_arch(arch)
        return self._compiler.compile_to_cubin(ptx_code, arch_enum)
    
    def compile(
        self,
        mlir_code: str,
        arch: str = 'sm90',
        opt_level: int = 2
    ) -> str:
        """
        Full compilation: MLIR → CUBIN.
        
        Args:
            mlir_code: MLIR source code
            arch: Target architecture
            opt_level: Optimization level
        
        Returns:
            Path to compiled CUBIN file
        """
        arch_enum = self._parse_arch(arch)
        return self._compiler.compile(mlir_code, arch_enum, opt_level)
    
    @staticmethod
    def _parse_arch(arch: str) -> int:
        """Convert arch string to enum value."""
        arch_map = {
            'sm80': 80, 'ampere': 80,
            'sm90': 90, 'hopper': 90,
            'sm100': 100, 'blackwell': 100
        }
        return arch_map.get(arch.lower(), 90)

#===----------------------------------------------------------------------===#
# Kernel Executor
#===----------------------------------------------------------------------===#

class Kernel:
    """Low-level kernel executor."""
    
    def __init__(self):
        self._executor = _cpp.KernelExecutor()
        self._kernel_set = False
    
    def load_cubin(self, cubin_path: Union[str, Path]):
        """Load compiled kernel from CUBIN file."""
        self._executor.load_cubin(str(cubin_path))
    
    def load_ptx(self, ptx_path: Union[str, Path]):
        """Load kernel from PTX file."""
        self._executor.load_ptx(str(ptx_path))
    
    def set_kernel(self, kernel_name: str):
        """Set kernel function to execute."""
        self._executor.set_kernel(kernel_name)
        self._kernel_set = True
    
    def launch(
        self,
        args: list,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_mem: int = 0
    ):
        """
        Launch kernel.
        
        Args:
            args: Kernel arguments (pointers)
            grid: Grid dimensions (x, y, z)
            block: Block dimensions (x, y, z)
            shared_mem: Dynamic shared memory size in bytes
        """
        if not self._kernel_set:
            raise RuntimeError("Kernel not set. Call set_kernel() first.")
        
        self._executor.launch(args, grid, block, shared_mem)
    
    def synchronize(self):
        """Wait for kernel completion."""
        self._executor.synchronize()

#===----------------------------------------------------------------------===#
# High-Level GEMM Interface
#===----------------------------------------------------------------------===#

class Gemm:
    """
    High-level GEMM executor.
    
    Supports:
    - Ampere (SM80): MMA.16816 (TF32, FP16)
    - Hopper (SM90): Warpgroup MMA + TMA
    - Blackwell (SM100): TCGEN05 MMA
    """
    
    def __init__(
        self,
        M: int,
        N: int,
        K: int,
        dtype_a: str = 'float16',
        dtype_b: str = 'float16',
        dtype_c: str = 'float32',
        arch: str = 'sm90',
        use_tma: bool = True
    ):
        """
        Initialize GEMM executor.
        
        Args:
            M, N, K: Matrix dimensions (C[M,N] = A[M,K] @ B[K,N])
            dtype_a, dtype_b, dtype_c: Data types
            arch: Target architecture (sm80, sm90, sm100)
            use_tma: Use Tensor Memory Accelerator (SM90+)
        """
        self.M, self.N, self.K = M, N, K
        self.dtype_a = np.dtype(dtype_a)
        self.dtype_b = np.dtype(dtype_b)
        self.dtype_c = np.dtype(dtype_c)
        self.arch = arch
        self.use_tma = use_tma and self._supports_tma()
        
        # Create C++ executor
        arch_enum = Compiler._parse_arch(arch)
        self._executor = _cpp.GemmExecutor(
            M, N, K, 
            self.dtype_a, self.dtype_b, self.dtype_c,
            arch_enum, self.use_tma
        )
        
        self._compiled = False
    
    def compile(self, mlir_code: str, opt_level: int = 2):
        """
        Compile GEMM kernel from MLIR.
        
        Args:
            mlir_code: MLIR source code
            opt_level: Optimization level
        """
        self._executor.compile_from_mlir(mlir_code, opt_level)
        self._compiled = True
    
    def load_compiled(self, cubin_path: Union[str, Path]):
        """Load pre-compiled GEMM kernel."""
        self._executor.load_compiled(str(cubin_path))
        self._compiled = True
    
    def __call__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Execute GEMM: C = A @ B
        
        Args:
            A: Input matrix [M, K]
            B: Input matrix [K, N]
            C: Output matrix [M, N] (optional, allocated if None)
        
        Returns:
            Result matrix C
        """
        if not self._compiled:
            raise RuntimeError("Kernel not compiled. Call compile() first.")
        
        # Validate shapes
        if A.shape != (self.M, self.K):
            raise ValueError(f"A shape mismatch: {A.shape} != ({self.M}, {self.K})")
        if B.shape != (self.K, self.N):
            raise ValueError(f"B shape mismatch: {B.shape} != ({self.K}, {self.N})")
        
        # Validate dtypes
        if A.dtype != self.dtype_a:
            A = A.astype(self.dtype_a)
        if B.dtype != self.dtype_b:
            B = B.astype(self.dtype_b)
        
        # Allocate output
        if C is None:
            C = np.zeros((self.M, self.N), dtype=self.dtype_c)
        
        # Execute
        self._executor.execute(A, B, C)
        
        return C
    
    def _supports_tma(self) -> bool:
        """Check if TMA is supported."""
        arch_num = Compiler._parse_arch(self.arch)
        return arch_num >= 90
    
    @staticmethod
    def get_optimal_tile_size(
        M: int, N: int, K: int, 
        arch: str = 'sm90'
    ) -> Tuple[int, int, int]:
        """Get recommended tile sizes for architecture."""
        arch_enum = Compiler._parse_arch(arch)
        return _cpp.GemmExecutor.get_optimal_tile_size(M, N, K, arch_enum)

#===----------------------------------------------------------------------===#
# Utility Functions
#===----------------------------------------------------------------------===#

def compile_mlir(
    mlir_code: str,
    arch: str = 'sm90',
    output_path: Optional[str] = None
) -> str:
    """
    Convenience function to compile MLIR to CUBIN.
    
    Args:
        mlir_code: MLIR source code
        arch: Target architecture
        output_path: Output CUBIN path (optional)
    
    Returns:
        Path to compiled CUBIN
    """
    compiler = Compiler()
    cubin_path = compiler.compile(mlir_code, arch)
    
    if output_path:
        import shutil
        shutil.move(cubin_path, output_path)
        return output_path
    
    return cubin_path
