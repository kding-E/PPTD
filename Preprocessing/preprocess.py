"""
Smart contract preprocessing module for handling Solidity code transformations.

This module provides functionality to preprocess Solidity smart contracts,
including handling for loops, if statements, function definitions, and modifiers.
"""

import os
import re
from typing import Dict, List, Any, Tuple

# Slither framework imports
from slither_imports import (
    Slither, CallExpression, SolidityCall, LowLevelCall, 
    HighLevelCall, InternalCall
)

from utils import write_to_file, get_output_file_path, extract_source_details

def extract_for_loop_changes(node: Dict[str, Any], source_code: str, changes: List[Dict[str, Any]]) -> None:
    """
    Extract and transform for loop constructs into while loops.
    
    Args:
        node: AST node representing a for loop
        source_code: Source code content
        changes: List to store transformation changes
    """
    start_for_pos, _ = extract_source_details(node['src'])
    start_for_body, len_for_body = extract_source_details(node['body']['src'])

    # Extract initialization expression
    init_expression = ''
    if node['initializationExpression'] is not None:
        init_src = extract_source_segment(node['initializationExpression']['src'], source_code)
        init_expression = f'\n{init_src};'
    
    # Extract condition
    condition_src = extract_source_segment(node['condition']['src'], source_code)

    # Extract loop expression
    loop_expression = None
    if node['loopExpression'] is not None:
        loop_expression = extract_source_segment(node['loopExpression']['src'], source_code)

    # Calculate header length for substitution
    for_header_len = start_for_body - start_for_pos
    changes.append({
        'op': 'sub',
        'pos': start_for_pos,
        'len': for_header_len
    })

    changes.append({
        'op': 'add',
        'pos': start_for_pos,
        'str': f'{init_expression}\nwhile ({condition_src})'
    })

    # Handle different body node types
    body_node_type = node['body']['nodeType']
    if body_node_type in ['ExpressionStatement', 'IfStatement']:
        _handle_simple_body(start_for_body, len_for_body, loop_expression, changes)
    elif body_node_type == 'Block':
        _handle_block_body(start_for_body, len_for_body, loop_expression, changes)
    else:
        raise ValueError(f'Unhandled for loop body type: {body_node_type}')


def _handle_simple_body(start_body: int, len_body: int, loop_expression: str, changes: List[Dict[str, Any]]) -> None:
    """Handle for loop body that's a simple statement."""
    end_body = start_body + len_body
    changes.append({
        'op': 'add',
        'pos': start_body,
        'str': '{\n'
    })

    if loop_expression is not None:
        changes.append({
            'op': 'add',
            'pos': end_body + 1,
            'str': f'\n{loop_expression};\n}}'
        })


def _handle_block_body(start_body: int, len_body: int, loop_expression: str, changes: List[Dict[str, Any]]) -> None:
    """Handle for loop body that's a block statement."""
    end_body = start_body + len_body

    if loop_expression is not None:
        changes.append({
            'op': 'add',
            'pos': end_body - 1,
            'str': f'\n{loop_expression};\n'
        })


def extract_source_segment(src_info: str, source_code: str) -> str:
    """
    Extract source code segment based on src info.
    
    Args:
        src_info: Source mapping information (format: "start:length")
        source_code: Complete source code
        
    Returns:
        Extracted source code segment
    """
    start_pos, length = extract_source_details(src_info)
    return source_code[start_pos:start_pos + length]





def extract_if_statement_changes(node: Dict[str, Any], source_code: str, changes: List[Dict[str, Any]]) -> None:
    """
    Extract and transform if statement constructs to ensure proper else blocks.
    
    Args:
        node: AST node representing an if statement
        source_code: Source code content
        changes: List to store transformation changes
    """
    # Handle true body
    true_body = node['trueBody']
    if true_body['nodeType'] == 'Block':
        pass  # Block statements don't need modification
    elif true_body['nodeType'] in [
        'ExpressionStatement', 'Throw', 'Return', 'VariableDeclarationStatement',
        'InlineAssembly', 'IfStatement', 'Continue', 'Break', 'EmitStatement'
    ]:
        start_true_body, len_true_body = extract_source_details(true_body['src'])
        _wrap_statement_in_block(start_true_body, len_true_body, source_code, changes)
    else:
        raise ValueError(f'Unhandled true body type: {true_body["nodeType"]}')

    # Handle false body
    false_body = node['falseBody']
    if false_body is not None:
        if false_body['nodeType'] in [
            'ExpressionStatement', 'Throw', 'Return', 'VariableDeclarationStatement', 'EmitStatement'
        ]:
            start_false_body, len_false_body = extract_source_details(false_body['src'])
            _wrap_statement_in_block(start_false_body, len_false_body, source_code, changes)
        elif 'condition' in false_body or false_body['nodeType'] == 'Block':
            return  # Already handled properly
        else:
            raise ValueError(f'Unhandled false body type: {false_body["nodeType"]}')
    else:
        # Add empty else block for if statements without else
        _add_empty_else_block(node, source_code, changes)


def _wrap_statement_in_block(start_body: int, len_body: int, source_code: str, changes: List[Dict[str, Any]]) -> None:
    """Wrap a single statement in block braces."""
    body_src = source_code[start_body:start_body + len_body + 1]
    changes.append({
        'op': 'sub',
        'pos': start_body,
        'len': len_body + 1
    })
    changes.append({
        'op': 'add',
        'pos': start_body,
        'str': f'{{\n{body_src}\n}}'
    })


def _add_empty_else_block(node: Dict[str, Any], source_code: str, changes: List[Dict[str, Any]]) -> None:
    """Add an empty else block to an if statement."""
    start_if, len_if = extract_source_details(node['src'])
    
    # Check if the last character is '}' to determine proper positioning
    if source_code[start_if + len_if - 1] != '}':
        end_if_body = start_if + len_if + 1
    else:
        end_if_body = start_if + len_if

    changes.append({
        'op': 'add',
        'pos': end_if_body,
        'str': ' else {\n\n}\n'
    })


# Remove the redundant extract_full_if_body function as its functionality is now in _wrap_statement_in_block


def extract_function_definition_changes(node: Dict[str, Any], source_code: str, changes: List[Dict[str, Any]]) -> None:
    """
    Remove view, constant, or pure modifiers from function definitions.
    
    Args:
        node: AST node representing a function definition
        source_code: Source code content
        changes: List to store transformation changes
    """
    state_mutability = node['stateMutability']
    
    if state_mutability in ['view', 'constant', 'pure']:
        start_func, len_func = extract_source_details(node['src'])
        func_src = source_code[start_func:start_func + len_func]

        # Find the mutability modifier in the function
        mutability_pattern = re.compile(r'\s*(view|constant|pure)\s*')
        matches = list(mutability_pattern.finditer(func_src))
        
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {state_mutability} modifier, found {len(matches)}')

        mutability_match = matches[0]
        mutability_start = mutability_match.start(1)
        mutability_len = mutability_match.end(1) - mutability_start

        changes.append({
            'op': 'sub',
            'pos': start_func + mutability_start,
            'len': mutability_len
        })
    elif state_mutability in ['nonpayable', 'payable']:
        pass  # No changes needed for these mutability types
    else:
        raise ValueError(f'Unhandled state mutability: {state_mutability}')

# Modifier processing functions
def collect_modifiers_info(ast_node: Any, source_code: str, modifiers: Dict[str, List[str]], 
                         modifier_definitions: Dict[str, str], function_sources: Dict[str, str]) -> None:
    """
    Collect information about modifiers and functions that use them.
    
    Args:
        ast_node: AST node to traverse
        source_code: Source code content
        modifiers: Dictionary mapping function names to their modifier names
        modifier_definitions: Dictionary mapping modifier names to their source code
        function_sources: Dictionary mapping function names to their source code
    """
    if isinstance(ast_node, dict):
        if 'nodeType' in ast_node:
            if ast_node['nodeType'] == 'ModifierDefinition':
                _extract_modifier_definition(ast_node, source_code, modifier_definitions)
            elif ast_node['nodeType'] == 'FunctionDefinition':
                _extract_function_with_modifiers(ast_node, source_code, modifiers, function_sources)
        
        # Recursively process child nodes
        for value in ast_node.values():
            collect_modifiers_info(value, source_code, modifiers, modifier_definitions, function_sources)
    
    elif isinstance(ast_node, list):
        for item in ast_node:
            collect_modifiers_info(item, source_code, modifiers, modifier_definitions, function_sources)


def _extract_modifier_definition(node: Dict[str, Any], source_code: str, modifier_definitions: Dict[str, str]) -> None:
    """Extract modifier definition source code."""
    src_parts = node['src'].split(':')
    start_pos = int(src_parts[0])
    length = int(src_parts[1])
    modifier_definitions[node['name']] = source_code[start_pos:start_pos + length]


def _extract_function_with_modifiers(node: Dict[str, Any], source_code: str, 
                                   modifiers: Dict[str, List[str]], function_sources: Dict[str, str]) -> None:
    """Extract function information if it uses modifiers."""
    if not node['modifiers']:
        return
    
    src_parts = node['src'].split(':')
    start_pos = int(src_parts[0])
    length = int(src_parts[1])
    function_sources[node['name']] = source_code[start_pos:start_pos + length]
    
    modifier_names = [mod['modifierName']['name'] for mod in node['modifiers']]
    modifiers[node['name']] = modifier_names
def refactor_modifiers(modifiers: Dict[str, List[str]], modifier_definitions: Dict[str, str], 
                      function_sources: Dict[str, str], source_code: str) -> str:
    """
    Refactor modifiers by inlining them into functions.
    Currently supports modifiers with and without parameters.
    
    Args:
        modifiers: Dictionary mapping function names to their modifier names
        modifier_definitions: Dictionary mapping modifier names to their source code
        function_sources: Dictionary mapping function names to their source code
        source_code: Complete source code
        
    Returns:
        Modified source code with inlined modifiers
    """
    modifier_calls_to_remove = []
    
    for function_name, modifier_names in modifiers.items():
        function_source = function_sources[function_name]
        function_body = _extract_function_body(function_source)
        original_function_body = function_body
        
        for modifier_name in modifier_names:
            if modifier_name not in modifier_definitions:
                continue
                
            modifier_source = modifier_definitions[modifier_name]
            
            # Handle modifiers with parameters
            modifier_with_params = _handle_modifier_parameters(
                modifier_name, modifier_source, function_source
            )
            
            if modifier_with_params:
                modifier_calls_to_remove.append(modifier_with_params)
                # Update modifier source with parameter substitution
                modifier_source = _substitute_modifier_parameters(
                    modifier_source, modifier_with_params, function_source
                )
            
            # Extract modifier body and inline it
            modifier_body = _extract_modifier_body(modifier_source)
            function_body = modifier_body.replace('_;', function_body)
        
        # Replace function body in source code
        new_function_source = function_source.replace(original_function_body, function_body)
        source_code = source_code.replace(function_source, new_function_source)
    
    # Remove modifier calls and definitions
    source_code = _cleanup_modifier_artifacts(
        source_code, modifier_calls_to_remove, modifier_definitions
    )
    
    return source_code


def _extract_function_body(function_source: str) -> str:
    """Extract the body content of a function (content between { and })."""
    match = re.search(r'^[^{]*{(.*)}$', function_source, re.DOTALL)
    if match:
        return match.group(1)
    return function_source


def _extract_modifier_body(modifier_source: str) -> str:
    """Extract the body content of a modifier (content between { and })."""
    match = re.search(r'^[^{]*{(.*)}$', modifier_source, re.DOTALL)
    if match:
        return match.group(1)
    return modifier_source


def _handle_modifier_parameters(modifier_name: str, modifier_source: str, function_source: str) -> str:
    """
    Handle modifier parameters by extracting parameter information.
    
    Returns:
        String representing modifier call with parameters, or empty string if no parameters
    """
    # Look for modifier call with parameters in function
    pattern = modifier_name + r'\([^\)]*\)'
    match = re.search(pattern, function_source)
    
    if match and len(match.group()) > len(modifier_name) + 2:  # Has parameters
        return match.group()
    
    return ''


def _substitute_modifier_parameters(modifier_source: str, modifier_call: str, function_source: str) -> str:
    """Substitute modifier parameters with actual values from function call."""
    # Extract parameter names from modifier definition
    modifier_params = _extract_parameter_names(modifier_source.split('\n')[0])
    
    # Extract parameter values from function call
    function_params = _extract_parameter_values(modifier_call)
    
    # Substitute parameters in modifier source
    modified_source = modifier_source
    for param_name, param_value in zip(modifier_params, function_params):
        modified_source = modified_source.replace(param_name, param_value)
    
    return modified_source


def _extract_parameter_names(definition_line: str) -> List[str]:
    """Extract parameter names from modifier definition line."""
    params = []
    # Find parameters between parentheses
    param_matches = re.findall(r'[^(^,]*,', definition_line)
    for match in param_matches:
        if len(match) > 2:
            # Extract identifier before comma
            identifier = re.search(r'[A-Za-z_][A-Za-z0-9_]*,', match)
            if identifier:
                params.append(identifier.group()[:-1])  # Remove comma
    
    # Handle last parameter (no comma)
    last_param = re.search(r'[A-Za-z_][A-Za-z0-9_]*\)', definition_line)
    if last_param and len(last_param.group()) > 1:
        params.append(last_param.group()[:-1])  # Remove closing parenthesis
    
    return params


def _extract_parameter_values(function_call: str) -> List[str]:
    """Extract parameter values from function modifier call."""
    params = []
    # Find parameters between parentheses
    param_matches = re.findall(r'[^(^,]*,', function_call)
    for match in param_matches:
        if len(match) > 2:
            params.append(match[:-1].strip())  # Remove comma and whitespace
    
    # Handle last parameter
    last_param = re.search(r'[^(^,]*\)', function_call)
    if last_param:
        params.append(last_param.group()[:-1].strip())  # Remove closing parenthesis
    
    return params


def _cleanup_modifier_artifacts(source_code: str, modifier_calls: List[str], 
                               modifier_definitions: Dict[str, str]) -> str:
    """Remove modifier calls and definitions from source code."""
    # Remove modifier calls from functions
    for modifier_call in modifier_calls:
        source_code = source_code.replace(modifier_call, ' ')
    
    # Remove modifier definitions
    for modifier_name, modifier_source in modifier_definitions.items():
        source_code = source_code.replace(modifier_source, '')
        # Remove various forms of modifier calls
        patterns_to_remove = [
            f'{modifier_name}()',
            f' {modifier_name} ',
            f'\n{modifier_name}\n',
            f'\n{modifier_name} ',
            f' {modifier_name}\n'
        ]
        for pattern in patterns_to_remove:
            source_code = source_code.replace(pattern, ' ')
    
    return source_code


def refactor_constructs(ast_node: Any, source_code: str, changes: List[Dict[str, Any]]) -> None:
    """
    Recursively traverse AST and refactor various language constructs.
    
    Args:
        ast_node: AST node to process
        source_code: Source code content
        changes: List to store transformation changes
    """
    if isinstance(ast_node, dict):
        if 'nodeType' in ast_node:
            node_type = ast_node['nodeType']
            
            if node_type == 'ForStatement':
                extract_for_loop_changes(ast_node, source_code, changes)
            elif node_type == 'IfStatement':
                extract_if_statement_changes(ast_node, source_code, changes)
            elif node_type == 'FunctionDefinition':
                extract_function_definition_changes(ast_node, source_code, changes)
        
        # Recursively process child nodes
        for value in ast_node.values():
            refactor_constructs(value, source_code, changes)
    
    elif isinstance(ast_node, list):
        for item in ast_node:
            refactor_constructs(item, source_code, changes)


def preprocess_contracts(file_name: str, output_dir: str) -> None:
    """
    Preprocess Solidity contracts by removing comments, handling modifiers, and transforming constructs.
    
    Args:
        file_name: Path to the source contract file
        output_dir: Output directory for processed files
    """
    # Remove comments and tabs
    output_file_name = replace_tabs_with_spaces(file_name, output_dir)
    
    with open(output_file_name, 'r', encoding='utf-8') as file:
        preprocessed_content = file.read()
    
    # Process modifiers
    slither = Slither(output_file_name, is_truffle=os.path.isdir(output_file_name))

    for source_file in slither.crytic_compile.asts:
        ast = slither.crytic_compile.asts[source_file]
        modifiers = {}  # Store function to modifier mappings
        modifier_definitions = {}  # Store modifier source code definitions
        function_sources = {}  # Store function source code
        source_code = slither.crytic_compile.src_content_for_file(source_file)
        original_source_code = source_code
        
        collect_modifiers_info(ast, source_code, modifiers, modifier_definitions, function_sources)
        source_code = refactor_modifiers(modifiers, modifier_definitions, function_sources, source_code)
        preprocessed_content = preprocessed_content.replace(original_source_code, source_code)
        
        # Process require statements
        preprocessed_content = replace_require_with_if(preprocessed_content)
        output_file_name = get_output_file_path(output_dir, output_file_name)
        
        write_to_file(output_file_name, 'w', preprocessed_content)
        
        # TODO: Remove hardcoded path - this should be configurable
        # shutil.copyfile(output_file_name, f'/home/dk/project/PPCD/instrument/contracts/other/temp/temp/aaa.sol')

    # Process language constructs
    slither = Slither(output_file_name, is_truffle=os.path.isdir(output_file_name))
    changes = []
    
    for source_file in slither.crytic_compile.asts:
        ast = slither.crytic_compile.asts[source_file]
        source_code = slither.crytic_compile.src_content_for_file(source_file)
        refactor_constructs(ast, source_code, changes)
        update_src_with_changes(changes, output_dir, source_code, source_file)

def replace_require_with_if(source_code: str) -> str:
    """
    Replace require statements with if-revert constructs.
    
    Args:
        source_code: Source code to transform
        
    Returns:
        Transformed source code with require statements replaced
    """
    # Remove error messages from require statements (for compatibility)
    source_code = re.sub(r'(require[^,]*)(,\s*"[^;]*);', r'\1);', source_code)
    
    # Transform require(condition); to if(!(condition)){ revert(); }else{ }
    source_code = re.sub(r'(require[^;]*)\);', r'\1)){ revert(); }else{ }', source_code)
    
    # Replace require( with if(!(
    source_code = re.sub(r'require[^(]*\(', 'if(!(', source_code)
    
    return source_code


def replace_tabs_with_spaces(file_name: str, output_dir: str) -> str:
    """
    Replace tabs with spaces and remove comments from source file.
    
    Args:
        file_name: Input file path
        output_dir: Output directory
        
    Returns:
        Path to the processed output file
    """
    with open(file_name, 'r', encoding='utf-8') as file:
        file_content = file.read()
    
    # Replace tabs with spaces
    file_content = re.sub(r'\t', ' ', file_content)
    
    # Remove single-line comments (// ...)
    file_content = re.sub(r'//[^\n]*', ' ', file_content)
    
    # Remove multi-line comments (/* ... */)
    file_content = re.sub(r'/\*[\w\W]*?\*/', ' ', file_content)
    
    output_file_name = get_output_file_path(output_dir, file_name)
    write_to_file(output_file_name, 'w', file_content)
    
    return output_file_name


def update_src_with_changes(changes: List[Dict[str, Any]], output_dir: str, 
                          source_code: str, source_file: str) -> None:
    """
    Apply accumulated changes to source code and write to output file.
    
    Args:
        changes: List of changes to apply
        output_dir: Output directory
        source_code: Original source code
        source_file: Source file path
    """
    # Group changes by position
    position_changes = {}
    for change in changes:
        pos = change['pos']
        if pos not in position_changes:
            position_changes[pos] = {'add': [], 'sub': []}
        
        operation = change['op']
        if 'add' in operation:
            position_changes[pos]['add'].append(change)
        else:
            position_changes[pos]['sub'].append(change)
    
    # Convert to sorted list
    grouped_changes = [
        {'pos': pos, 'val': operations} 
        for pos, operations in position_changes.items()
    ]
    grouped_changes.sort(key=lambda x: x['pos'])
    
    # Apply changes to source code
    updated_source = _apply_changes_to_source(source_code, grouped_changes)
    
    # Write updated source to output file
    output_file_name = get_output_file_path(output_dir, source_file)
    write_to_file(updated_source, output_file_name, 'w')


def _apply_changes_to_source(source_code: str, grouped_changes: List[Dict[str, Any]]) -> str:
    """Apply grouped changes to source code in order."""
    source_length = len(source_code)
    current_pos = 0
    updated_source = ''
    
    for change_group in grouped_changes:
        change_pos = change_group['pos']
        change_operations = change_group['val']
        
        # Add unchanged content before current position
        updated_source += source_code[current_pos:change_pos]
        current_pos = change_pos
        
        # Handle substitution operations
        sub_changes = change_operations['sub']
        if sub_changes:
            # Find maximum length to substitute
            max_length = max(change['len'] for change in sub_changes)
            current_pos += max_length
        
        # Handle addition operations
        add_changes = change_operations['add']
        for add_change in add_changes:
            updated_source += add_change['str']
    
    # Add remaining source code
    if current_pos < source_length:
        updated_source += source_code[current_pos:]
    
    return updated_source


def separate_multiple_called_functions(slither: Slither, output_dir: str) -> None:
    """
    Handle multiple calls to the same function by creating duplicate functions.
    
    Args:
        slither: Slither instance
        output_dir: Output directory
    """
    source_path_to_instructions = {}
    _collect_instruction_mappings(slither, source_path_to_instructions)
    _sort_and_merge_instructions(source_path_to_instructions)
    
    source_path_to_code = {}
    _generate_instrumented_code(slither, source_path_to_instructions, source_path_to_code)
    
    # Write instrumented code to files
    for source_path, source_code in source_path_to_code.items():
        output_path = get_output_file_path(output_dir, source_path)
        write_to_file(output_path, 'w', source_code)


def _collect_instruction_mappings(slither: Slither, source_path_to_instructions: Dict[str, List]) -> None:
    """Collect function call details and generate instruction mappings."""
    called_contracts = {}
    _analyze_function_calls(called_contracts, slither)
    _populate_destination_instructions(called_contracts, source_path_to_instructions, slither)
    _populate_source_instructions(called_contracts, source_path_to_instructions)


def _populate_source_instructions(called_contracts: Dict, source_path_to_instructions: Dict[str, List]) -> None:
    """Populate instructions at function call sites."""
    for contract, called_functions in called_contracts.items():
        for function, callers in called_functions.items():
            if len(callers) <= 1:
                continue
                
            for index, caller in enumerate(callers):
                if index == 0:
                    continue  # Skip the first caller
                    
                start_pos = caller.called.source_mapping["start"]
                length = caller.called.source_mapping["length"]
                injection_pos = start_pos + length
                
                contract_file_name = contract.source_mapping['filename_used']
                
                if contract_file_name not in source_path_to_instructions:
                    source_path_to_instructions[contract_file_name] = []
                
                instruction = {
                    "pos": injection_pos,
                    "instr": f'_{index}'
                }
                source_path_to_instructions[contract_file_name].append(instruction)


def _analyze_function_calls(called_contracts: Dict, slither: Slither) -> None:
    """
    Analyze function calls throughout the contract to identify duplicate calls.
    
    Args:
        called_contracts: Dictionary to store contract call mappings
        slither: Slither instance
    """
    for contract in slither.contracts:
        if contract.is_interface:
            raise ValueError(f'Interface contract {contract.name} is not supported')

        for function in contract.functions + contract.modifiers:
            if function.is_constructor_variables:
                continue

            for node in function.nodes:
                for ir in node.irs_ssa:
                    if isinstance(ir, SolidityCall):
                        continue  # Skip Solidity built-in calls
                    elif isinstance(ir, LowLevelCall):
                        continue  # Skip low-level calls
                    elif isinstance(ir, (HighLevelCall, InternalCall)):
                        _process_function_call(ir, called_contracts)


def _process_function_call(ir, called_contracts: Dict) -> None:
    """Process a single function call and update called_contracts mapping."""
    called_contract = ir.function.contract
    if called_contract not in called_contracts:
        called_contracts[called_contract] = {}

    called_function = ir.function
    if called_function not in called_contracts[called_contract]:
        called_contracts[called_contract][called_function] = set()

    expression = ir.expression
    if isinstance(expression, CallExpression):
        called_contracts[called_contract][called_function].add(expression)


def _populate_destination_instructions(called_contracts: Dict, source_path_to_instructions: Dict[str, List], 
                                     slither: Slither) -> None:
    """
    Generate duplicate functions and add them to instruction mappings.
    
    Args:
        called_contracts: Dictionary containing function call mappings
        source_path_to_instructions: Dictionary to store instruction mappings
        slither: Slither instance
    """
    for contract, called_functions in called_contracts.items():
        duplicate_functions = []
        
        for function, callers in called_functions.items():
            if len(callers) <= 1:
                continue
                
            # Generate duplicate functions
            contract_file_name = contract.source_mapping['filename_used']
            contract_source_code = slither.crytic_compile.src_content[contract_file_name]
            
            start_pos = function.source_mapping["start"]
            length = function.source_mapping["length"]
            function_source = contract_source_code[start_pos:start_pos + length]
            
            function_prefix = f'function {function.name}'
            
            for index, caller in enumerate(callers):
                if index == 0:
                    continue  # Skip the first caller
                    
                duplicate_function = re.sub(
                    f'^{function_prefix}', 
                    f'{function_prefix}_{index}', 
                    function_source
                )
                duplicate_functions.append(duplicate_function)
            
            # Add duplicate functions to instructions
            if duplicate_functions:
                duplicate_functions_str = '\n' + '\n\n'.join(duplicate_functions) + '\n'
                contract_start = contract.source_mapping["start"]
                contract_length = contract.source_mapping["length"]
                injection_pos = contract_start + contract_length - 1
                
                if contract_file_name not in source_path_to_instructions:
                    source_path_to_instructions[contract_file_name] = []
                
                instruction = {
                    "pos": injection_pos,
                    "instr": duplicate_functions_str
                }
                source_path_to_instructions[contract_file_name].append(instruction)


def _sort_and_merge_instructions(source_path_to_instructions: Dict[str, List]) -> None:
    """
    Sort instructions by position and merge instructions at the same position.
    
    Args:
        source_path_to_instructions: Dictionary containing instruction mappings
    """
    for source_path, instructions in source_path_to_instructions.items():
        # Sort instructions by position
        instructions.sort(key=lambda instruction: instruction['pos'])
        
        if len(instructions) < 2:
            continue
        
        # Merge instructions at the same position
        merged_instructions = []
        current_position = -1
        current_instruction_group = []
        
        for instruction in instructions:
            position = instruction["pos"]
            
            if current_position == -1:
                current_position = position
                current_instruction_group = [instruction["instr"]]
            elif current_position == position:
                current_instruction_group.append(instruction["instr"])
            else:
                # Save the previous group
                merged_instruction = {
                    "pos": current_position,
                    "instr": '\n\n'.join(current_instruction_group)
                }
                merged_instructions.append(merged_instruction)
                
                # Start a new group
                current_position = position
                current_instruction_group = [instruction["instr"]]
        
        # Don't forget the last group
        if current_instruction_group:
            merged_instruction = {
                "pos": current_position,
                "instr": '\n\n'.join(current_instruction_group)
            }
            merged_instructions.append(merged_instruction)
        
        source_path_to_instructions[source_path] = merged_instructions


def _generate_instrumented_code(slither: Slither, source_path_to_instructions: Dict[str, List], 
                              source_path_to_code: Dict[str, str]) -> None:
    """
    Generate instrumented source code by applying instructions.
    
    Args:
        slither: Slither instance
        source_path_to_instructions: Dictionary containing instruction mappings
        source_path_to_code: Dictionary to store generated code
    """
    # Get all source file names
    source_file_names = set()
    for contract in slither.contracts:
        contract_file_name = contract.source_mapping['filename_used']
        source_file_names.add(contract_file_name)
    
    # Process each source file
    for source_file_name in source_file_names:
        original_source = slither.crytic_compile.src_content[source_file_name]
        
        if source_file_name not in source_path_to_instructions:
            # No instructions for this file, use original source
            source_path_to_code[source_file_name] = original_source
            continue
        
        instructions = source_path_to_instructions[source_file_name]
        
        # Apply instructions to source code
        modified_source = _apply_instructions_to_source(original_source, instructions)
        source_path_to_code[source_file_name] = modified_source


def _apply_instructions_to_source(source_code: str, instructions: List[Dict[str, Any]]) -> str:
    """Apply instructions to source code at specified positions."""
    current_pos = 0
    modified_source = ""
    instruction_index = 0
    
    current_instruction = instructions[instruction_index] if instructions else None
    
    while current_pos < len(source_code):
        # Check if we need to apply an instruction at this position
        if current_instruction is not None and current_pos == current_instruction["pos"]:
            modified_source += current_instruction["instr"]
            
            # Move to next instruction
            instruction_index += 1
            current_instruction = instructions[instruction_index] if instruction_index < len(instructions) else None
        
        # Add the current character
        modified_source += source_code[current_pos]
        current_pos += 1
    
    return modified_source
