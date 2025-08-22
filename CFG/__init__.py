"""
CFG (Control Flow Graph) Analysis Package
Contains control flow graph analysis, path profiling, code instrumentation and other core functionalities

This package has been thoroughly cleaned and standardized with the following features:
- Unified code style and documentation standards
- Eliminated redundant file operation functions
- Standardized import statements and error handling
- Optimized module structure and API design
"""

# 导入核心类和函数，提供便捷的包级别访问
from .cfg_analyzer import CFGAnalyzer
from .contract_graph import ContractGraph, ContractGraphWithExitEntry, Edge, entry_node, exit_node

# 版本信息
__version__ = "1.0.0"
__author__ = "PPTD Team"

# 导出的公共接口
__all__ = [
    # 控制流分析
    'CFGAnalyzer',
    
    # 图结构
    'ContractGraph',
    'ContractGraphWithExitEntry', 
    'Edge',
    'entry_node',
    'exit_node',
]
