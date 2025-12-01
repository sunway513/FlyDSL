"""Utilities forwarding to shared tests/utils.py for backward compatibility."""

import sys
import os
import importlib.util

# Load the shared utils module from tests/utils.py
_utils_path = os.path.join(os.path.dirname(__file__), '../../utils.py')
_spec = importlib.util.spec_from_file_location("_shared_utils", _utils_path)
_shared_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_shared_utils)

# Re-export all utilities
compile_to_hsaco = _shared_utils.compile_to_hsaco
BenchmarkResults = _shared_utils.BenchmarkResults
perftest = _shared_utils.perftest

__all__ = ['compile_to_hsaco', 'BenchmarkResults', 'perftest']
