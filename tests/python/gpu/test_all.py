#!/usr/bin/env python3
"""
RocDSL GPU Test Suite - Comprehensive Test Runner

This script runs all GPU tests in sequence and reports results.
Tests cover:
- Basic GPU operations (vector add)
- Shared memory optimization
- MFMA matrix multiply-accumulate instructions
- Rocir coordinate operations and layouts
- Advanced matrix operations with custom layouts
"""
import sys
import subprocess
import os
from pathlib import Path

# Test configurations: (filename, description, critical)
TESTS = [
    ("test_gpu_simple.py", "Basic GPU Vector Addition", True),
    ("test_gpu_layout.py", "Tiled Layout Operations", True),
    ("test_shared_working.py", "Shared Memory Matmul", True),
    ("test_mfma_simple_working.py", "MFMA Instructions", True),
    ("test_gpu_with_rocir_coords.py", "Rocir Coordinate Lowering", True),
    ("test_gpu_rocdsl.py", "Advanced Rocir Layout Tests", True),
]

def run_test(test_file, description):
    """Run a single test and return (success, output)."""
    test_path = Path(__file__).parent / test_file
    
    if not test_path.exists():
        return False, f"Test file not found: {test_file}"
    
    try:
        result = subprocess.run(
            ["python3", str(test_path)],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        return success, output
    except subprocess.TimeoutExpired:
        return False, "Test timed out (60s limit)"
    except Exception as e:
        return False, f"Test execution failed: {e}"

def main():
    """Run all tests and generate report."""
    print("=" * 80)
    print("RocDSL GPU Test Suite")
    print("=" * 80)
    print()
    
    results = []
    passed = 0
    failed = 0
    
    for test_file, description, critical in TESTS:
        print(f"Running: {description}")
        print(f"  File: {test_file}")
        
        success, output = run_test(test_file, description)
        
        if success:
            print(f"  ✓ PASSED")
            passed += 1
            status = "PASS"
        else:
            print(f"  ✗ FAILED")
            failed += 1
            status = "FAIL"
            # Print last 20 lines of output for failures
            lines = output.strip().split('\n')
            print("\n  Last 20 lines of output:")
            for line in lines[-20:]:
                print(f"    {line}")
        
        print()
        results.append((test_file, description, status, critical, output))
    
    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    for test_file, description, status, critical, _ in results:
        marker = "✓" if status == "PASS" else "✗"
        critical_marker = " [CRITICAL]" if critical else ""
        print(f"{marker} {description}{critical_marker}")
        print(f"  {test_file}: {status}")
    
    print()
    print("=" * 80)
    print(f"Total: {passed + failed} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    # Check critical failures
    critical_failures = [
        (f, d) for f, d, s, c, _ in results 
        if s == "FAIL" and c
    ]
    
    if critical_failures:
        print()
        print("⚠ CRITICAL FAILURES:")
        for test_file, description in critical_failures:
            print(f"  - {description} ({test_file})")
        return 1
    
    if failed > 0:
        print()
        print("⚠ Some tests failed, but no critical failures")
        return 1
    
    print()
    print("✓ ALL TESTS PASSED!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
