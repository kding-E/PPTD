
"""
Inline Assembly Instrumentation Module

Provides efficient Solidity inline assembly instrumentation functionality without external contract dependencies.
Supports direct memory operations and logging for EPP and PAP algorithms.

"""

# ===============================================
# Import Core Classes and Functions
# ===============================================

# Import from instrument.py
from .instrumenter import (
    EPPIns,
    PAPIns, 
    EncodeIns,
    CodeInstrumenter,
    create_instrumenter,
    create_instrument
)

# Import from ins_tex.py
from .ins_tex import instrument_solidity_code

# Import from ins_bytecode.py
from .ins_bytecode import insert_bytecode_before_exits

# ===============================================
# Convenience Functions
# ===============================================

def get_inline_assembly_instrumentation(instr_type, *args, **kwargs):
    """
    Get inline assembly instrumentation code (convenience wrapper)
    
    Args:
        instr_type: Instrumentation type
        *args: Positional arguments
        **kwargs: Keyword arguments including 'paths', 'slither', 'memory_address'
    
    Returns:
        str: Inline assembly instrumentation code
    """
    # Create a default instrumenter if not provided
    if 'instrumenter' in kwargs:
        instrumenter = kwargs['instrumenter']
    else:
        paths = kwargs.get('paths', {})
        slither = kwargs.get('slither')
        memory_address = kwargs.get('memory_address', '0xa0')
        instrumenter = create_instrumenter(paths, slither, memory_address)
    
    return instrumenter.get_inline_assembly_instrumentation(instr_type, *args)

# ===============================================
# Exported Main Interfaces
# ===============================================

__all__ = [
    # Core instrumentation classes
    'EPPIns',
    'PAPIns', 
    'EncodeIns',
    'CodeInstrumenter',
    
    # Factory functions
    'create_instrumenter',
    'create_instrument',
    
    # Convenience functions
    'get_inline_assembly_instrumentation',
    
    # Specialized instrumentation functions
    'instrument_solidity_code',
    'insert_bytecode_before_exits'
]

# ===============================================
# Module metadata
# ===============================================

__version__ = '1.0.0'
__author__ = 'PPTD Team'



