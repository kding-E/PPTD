"""
Smart Contract Preprocessing Module

This module provides comprehensive preprocessing functionality for Solidity smart contracts,
including code transformation, comment removal, modifier handling, and language construct refactoring.

Main Features:
- Remove comments and normalize whitespace
- Transform for loops to while loops
- Handle if statements and ensure proper else blocks
- Inline modifiers into functions
- Remove view/pure/constant modifiers
- Replace require statements with if-revert constructs
- Handle multiple function calls with duplication
- AST-based source code transformations

Usage:
    from Preprocessing import preprocess_contracts
    
    # Preprocess a contract file
    preprocess_contracts('contract.sol', 'output_dir/')
"""

# ===============================================
# Import Main Functions
# ===============================================

# Import from preprocess.py
from .preprocess import (
    # Main preprocessing function
    preprocess_contracts,
    
    # Source code transformation functions
    replace_require_with_if,
    replace_tabs_with_spaces,
    
    # For loop handling
    extract_for_loop_changes,
    extract_source_segment,
    
    # If statement handling  
    extract_if_statement_changes,
    
    # Function definition handling
    extract_function_definition_changes,
    
    # Modifier processing
    collect_modifiers_info,
    refactor_modifiers,
    
    # Language construct refactoring
    refactor_constructs,
    
    # Source code update utilities
    update_src_with_changes,
    
    # Multiple function call handling
    separate_multiple_called_functions
)

# ===============================================
# Convenience Functions
# ===============================================

def preprocess_solidity_file(input_file: str, output_dir: str) -> None:
    """
    Convenience function to preprocess a single Solidity file
    
    Args:
        input_file: Path to the input Solidity file
        output_dir: Directory for preprocessed output
    """
    preprocess_contracts(input_file, output_dir)

def clean_solidity_code(source_code: str) -> str:
    """
    Clean Solidity code by removing comments and normalizing whitespace
    
    Args:
        source_code: Source code to clean
        
    Returns:
        Cleaned source code
    """
    import re
    
    # Replace tabs with spaces
    source_code = re.sub(r'\t', ' ', source_code)
    
    # Remove single-line comments (// ...)
    source_code = re.sub(r'//[^\n]*', ' ', source_code)
    
    # Remove multi-line comments (/* ... */)
    source_code = re.sub(r'/\*[\w\W]*?\*/', ' ', source_code)
    
    return source_code

def transform_require_statements(source_code: str) -> str:
    """
    Transform require statements to if-revert constructs
    
    Args:
        source_code: Source code to transform
        
    Returns:
        Transformed source code
    """
    return replace_require_with_if(source_code)

# ===============================================
# Exported Main Interfaces
# ===============================================

__all__ = [
    # Main preprocessing functions
    'preprocess_contracts',
    'preprocess_solidity_file',
    
    # Source code transformation functions
    'replace_require_with_if',
    'replace_tabs_with_spaces',
    'clean_solidity_code',
    'transform_require_statements',
    
    # Language construct handlers
    'extract_for_loop_changes',
    'extract_if_statement_changes', 
    'extract_function_definition_changes',
    'extract_source_segment',
    
    # Modifier processing
    'collect_modifiers_info',
    'refactor_modifiers',
    
    # AST refactoring
    'refactor_constructs',
    
    # Source code utilities
    'update_src_with_changes',
    
    # Advanced features
    'separate_multiple_called_functions'
]

# ===============================================
# Module metadata
# ===============================================

__version__ = '1.0.0'
__author__ = 'PPTD Team'
__description__ = 'Smart contract preprocessing and transformation utilities'