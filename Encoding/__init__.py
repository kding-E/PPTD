"""
Encoding Module - Encoding and Path Profiling Module

This module is the core encoding module of the PPTD (Private Path Detection) system, containing:
- Path profiler (path_profiler) - Supports EPP and PAP algorithms
- Code instrumentation submodule (Instrumentation) - Contains all instrumentation-related functionality

Main Features:
1. Efficient Path Profiling (EPP)
2. Profiling All Paths (PAP) 
3. Unified path profiling interface
4. Complete code instrumentation support

Architecture Design:
- PathProfiler: Unified path profiler class supporting multiple algorithms
- ProfilerType: Path profiler type enumeration
- Instrumentation: Code instrumentation submodule

Usage:
1. Direct import from submodules:
   from instrument.Encoding.Instrumentation import insert_bytecode
   
2. Using path profiling (requires CFG module support):
   from instrument.Encoding.path_profiler import PathProfiler, ProfilerType
"""

# Import instrumentation submodule (no external dependencies)
from . import Instrumentation

# Convenient access to instrumentation functionality (dependency-free basic functionality)
from .Instrumentation import (
    EPPIns,
    PAPIns,
    EncodeIns,
    create_instrumenter,
    get_inline_assembly_instrumentation
)

# Import other available functions
try:
    from .Instrumentation.ins_tex import instrument_solidity_code
except ImportError:
    pass

# Lazy import path profiling functionality (avoids CFG module dependency issues)
def get_path_profiler():
    """Lazy import PathProfiler to avoid dependency issues at startup."""
    try:
        from .path_profiler import PathProfiler
        return PathProfiler
    except ImportError as e:
        raise ImportError(f"Cannot import PathProfiler, please ensure CFG module is available: {e}")

def get_profiler_type():
    """Lazy import ProfilerType enumeration"""
    try:
        from .path_profiler import ProfilerType
        return ProfilerType
    except ImportError as e:
        raise ImportError(f"Cannot import ProfilerType, please ensure CFG module is available: {e}")

# Basic exports (no external dependencies)
__all__ = [
    # Instrumentation sub-module
    'Instrumentation',
    
    # Direct access to instrumentation functions (no dependencies)
    'insert_bytecode',
    'instrument_solidity_code',
    
    # Lazy import instrumentation functions (with dependencies)
    'get_instrumentation_types',
    'get_code_instrument',
    
    # Path profiling lazy imports (with dependencies)
    'get_path_profiler',
    'get_profiler_type',
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'PPTD Team'
__description__ = 'Encoding and path profiling module for PPTD system'

# Try to get supported algorithm types (if CFG module is available)
try:
    ProfilerType = get_profiler_type()
    SUPPORTED_ALGORITHMS = [
        ProfilerType.EPP,  # Efficient Path Profiling
        ProfilerType.PAP   # Path-aware Profiling
    ]
except ImportError:
    SUPPORTED_ALGORITHMS = []  # Empty list when CFG module is unavailable