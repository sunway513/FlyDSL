"""Shared utilities for GPU testing, compilation, and benchmarking."""

from rocdsl.compiler.pipeline import Pipeline, run_pipeline
from rocdsl.compiler.rocir_opt_helper import apply_rocir_coord_lowering
from rocdsl.runtime.hip_util import get_hip_arch
from hip import hip
import numpy as np
from functools import wraps


def compile_to_hsaco(mlir_module):
    """
    Compile MLIR module to HSACO binary for AMD GPUs.
    
    Pipeline:
    1. Apply rocir coordinate lowering (rocir ops -> arithmetic)
    2. Canonicalize and CSE
    3. Attach ROCDL target for current GPU architecture
    4. Convert GPU dialect to ROCDL
    5. Lower to LLVM
    6. Generate binary
    
    Args:
        mlir_module: MLIR module containing GPU kernels
        
    Returns:
        bytes: HSACO binary object
    """
    # Apply rocir coordinate lowering first
    lowered_module = apply_rocir_coord_lowering(mlir_module)
    
    # Get the current GPU architecture
    gpu_arch = get_hip_arch()
    
    # Then run the main GPU compilation pipeline
    lowered = run_pipeline(
        lowered_module,
        Pipeline()
        .canonicalize()
        .cse()
        .rocdl_attach_target(chip=gpu_arch)
        .Gpu(Pipeline().convert_gpu_to_rocdl(use_bare_ptr_memref_call_conv=True, runtime="HIP"))
        .gpu_to_llvm()
        .lower_to_llvm()
        .gpu_module_to_binary(format="bin")
    )
    from rocdsl.dialects.ext.gpu import get_compile_object_bytes
    return get_compile_object_bytes(lowered)


class BenchmarkResults:
    """Container for benchmark timing results."""
    
    def __init__(self, times_ms, size, dtype_bytes=4):
        """
        Initialize benchmark results.
        
        Args:
            times_ms: List of execution times in milliseconds
            size: Number of elements processed
            dtype_bytes: Size of each element in bytes (default: 4 for float32)
        """
        self.times_ms = times_ms
        self.size = size
        self.dtype_bytes = dtype_bytes
        
    @property
    def avg_ms(self):
        """Average execution time in milliseconds."""
        return np.mean(self.times_ms)
    
    @property
    def min_ms(self):
        """Minimum execution time in milliseconds."""
        return np.min(self.times_ms)
    
    @property
    def max_ms(self):
        """Maximum execution time in milliseconds."""
        return np.max(self.times_ms)
    
    @property
    def std_ms(self):
        """Standard deviation of execution times in milliseconds."""
        return np.std(self.times_ms)
    
    @property
    def bandwidth_gbs(self):
        """Calculate achieved bandwidth in GB/s (assuming 3x memory traffic: 2 reads + 1 write)."""
        # For vector operations like A + B = C, we have 3 memory operations
        total_bytes = 3 * self.size * self.dtype_bytes
        return (total_bytes / 1e9) / (self.avg_ms / 1000)
    
    def __str__(self):
        """Format benchmark results as a string."""
        result = "  Benchmark Results:\n"
        result += f"  Average Time:  {self.avg_ms:.3f} ms\n"
        result += f"  Min Time:      {self.min_ms:.3f} ms\n"
        result += f"  Max Time:      {self.max_ms:.3f} ms\n"
        result += f"  Std Dev:       {self.std_ms:.3f} ms\n"
        result += f"  Bandwidth:     {self.bandwidth_gbs:.2f} GB/s"
        return result


def perftest(func):
    """
    Decorator for benchmarking GPU kernels.
    
    The decorated function should return a tuple containing:
        (kernel_func, args, grid_dims, block_dims, size)
    
    The decorator will:
    1. Run warmup iterations
    2. Run benchmark iterations with timing
    3. Return BenchmarkResults object
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get kernel configuration from the decorated function
        kernel_func, kernel_args, grid_dims, block_dims, size = func(*args, **kwargs)
        
        # Warmup iterations
        warmup_iters = 5
        print(f"\n  Running {warmup_iters} warmup iterations...")
        for i in range(warmup_iters):
            hip.hipModuleLaunchKernel(
                kernel_func,
                *grid_dims,
                *block_dims,
                sharedMemBytes=0,
                stream=None,
                kernelParams=kernel_args,
                extra=None
            )
        hip.hipDeviceSynchronize()
        
        # Benchmark iterations
        num_iters = 100
        print(f"  Running {num_iters} benchmark iterations...")
        
        times_ms = []
        for i in range(num_iters):
            # Create events for timing
            start_event = hip.hipEventCreate()[1]
            stop_event = hip.hipEventCreate()[1]
            
            # Record start event
            hip.hipEventRecord(start_event, None)
            
            # Launch kernel
            hip.hipModuleLaunchKernel(
                kernel_func,
                *grid_dims,
                *block_dims,
                sharedMemBytes=0,
                stream=None,
                kernelParams=kernel_args,
                extra=None
            )
            
            # Record stop event
            hip.hipEventRecord(stop_event, None)
            
            # Wait for completion
            hip.hipEventSynchronize(stop_event)
            
            # Get elapsed time
            elapsed_ms = hip.hipEventElapsedTime(start_event, stop_event)[1]
            times_ms.append(elapsed_ms)
            
            # Cleanup events
            hip.hipEventDestroy(start_event)
            hip.hipEventDestroy(stop_event)
        
        return BenchmarkResults(times_ms, size)
    
    return wrapper

