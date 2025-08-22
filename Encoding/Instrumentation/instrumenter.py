"""
Unified Code Instrumentation Module

Integrates all instrumentation-related functionalities including code instrumentation
for EPP and PAP algorithms, position calculation, and source code injection.

"""

# Slither 框架统一导入
from slither_imports import NodeType
from CFG.contract_graph import exit_node, entry_node
from utils import  get_children, get_unique_node_id, write_to_file

"""
Inline Assembly Instrumentation Module

Provides efficient Solidity inline assembly instrumentation functionality without external contract dependencies.
Supports direct memory operations and logging for EPP and PAP algorithms.

"""

# ===============================================
# Inline Assembly Instrumentation Classes (Efficient, No Dependencies)
# ===============================================

class EPPIns:
    """EPP Algorithm Inline Assembly Instrumenter"""
    
    def __init__(self, memory_address='0xa0'):
        """
        Initialize EPP instrumenter
        
        Args:
            memory_address: Memory address (default 0xa0)
        """
        self.memory_address = memory_address

    def get_val_init(self, counter_init_val):
        """Initialize data counter"""
        assert counter_init_val != 0
        return f'assembly {{ mstore({self.memory_address}, {counter_init_val}) }}'

    def get_epp_counter(self, counter_inc):
        """Increment counter"""
        assert counter_inc != 0
        return f'assembly {{ mstore({self.memory_address}, add(mload({self.memory_address}), {counter_inc})) }}'

    def get_epp_counter_and_flush(self, counter_inc):
        """Increment and flush path count"""
        if counter_inc == 0:
            return f'assembly {{ log0({self.memory_address},0x20) mstore({self.memory_address}, 0) }}'
        else:
            return f'assembly {{ mstore({self.memory_address}, add(mload({self.memory_address}), {counter_inc})) log0({self.memory_address},0x20) mstore({self.memory_address}, 0) }}'

    def get_epp_reset_counter(self, counter_reset):
        """Reset counter"""
        return f'assembly {{ mstore({self.memory_address}, {counter_reset}) }}'


class PAPIns:
    """PAP Algorithm Inline Assembly Instrumenter"""
    
    def __init__(self, memory_address='0xa0'):
        """
        Initialize PAP instrumenter
        
        Args:
            memory_address: Memory address (default 0xa0)
        """
        self.memory_address = memory_address
    
    def get_val_init(self, counter_init_val):
        """Initialize data counter"""
        assert counter_init_val != 0
        return f'assembly {{ mstore({self.memory_address}, {counter_init_val}) }}'

    def get_pap_val(self, last_node_id):
        """Output path ID"""
        return f'assembly {{ log0({self.memory_address},0x20) }}'

    def get_pap_counter(self, s, i):
        """PAP algorithm r counter operation"""
        return f'assembly {{ mstore({self.memory_address}, add(mul(mload({self.memory_address}), {s}), {i})) }}'


class EncodeIns:
    """Solidity Code Inline Assembly Instrumenter"""
    
    def __init__(self, paths, slither, memory_address='0xa0'):
        """
        Initialize instrumenter
        
        Args:
            paths: Path information dictionary
            slither: Slither analysis instance
            memory_address: Memory address (default 0xa0)
        """
        self.slither = slither
        self.memory_address = memory_address
        self.path_statements_contain_loop = False
        
        # Check if loop paths are included
        for pid in paths:
            path_details = paths[pid]
            is_loop_path = path_details['is_loop_path']
            if is_loop_path:
                self.path_statements_contain_loop = True
                break

        # Select instrumentation strategy: PAP for loops, EPP for non-loops
        if self.path_statements_contain_loop:
            self.instr = PAPIns(self.memory_address)
        else:
            self.instr = EPPIns(self.memory_address)

    def get_inline_assembly_instrumentation(self, instr_type, *args):
        """
        Get inline assembly instrumentation code
        
        Args:
            instr_type: Instrumentation type
                - 'init': Initialize counter
                - 'inc_counter': Increment counter  
                - 'flush': Flush and output path data
                - 'reset': Reset counter
                - 'pap_counter': PAP algorithm r counter operation
            *args: Parameters for corresponding instrumentation method
            
        Returns:
            str: Inline assembly instrumentation code
        """
        if instr_type == 'init':
            return self.instr.get_val_init(*args)
        elif instr_type == 'inc_counter':
            return self.instr.get_epp_counter(*args)
        elif instr_type == 'flush':
            if hasattr(self.instr, 'get_epp_counter_and_flush'):
                return self.instr.get_epp_counter_and_flush(*args)
            else:
                return self.instr.get_pap_val(*args)
        elif instr_type == 'reset':
            return self.instr.get_epp_reset_counter(*args)
        elif instr_type == 'pap_counter':
            return self.instr.get_pap_counter(*args)
        else:
            raise ValueError(f"Unknown instrumentation type: {instr_type}")

    def is_using_loops(self):
        """Return whether using loop paths (PAP algorithm)"""
        return self.path_statements_contain_loop

    def get_instrumentation_strategy(self):
        """Return name of current instrumentation strategy"""
        return "PAP" if self.path_statements_contain_loop else "EPP"


# ===============================================
# Factory Functions and Convenience Interfaces
# ===============================================

def create_instrumenter(paths, slither, memory_address='0xa0'):
    """
    Create inline assembly instrumenter
    
    Args:
        paths: Path information dictionary
        slither: Slither analysis instance
        memory_address: Memory address (default 0xa0, customizable like 0x80, 0xc0, etc.)
        
    Returns:
        EncodeIns: Configured inline assembly instrumenter
    """
    return EncodeIns(paths, slither, memory_address)




class CodeInstrumenter:
    """Unified code instrumenter"""

    def __init__(self):
        """Initialize the instrumenter with inline assembly mode"""
        self.formatters = self._initialize_formatters()
    
    def _initialize_formatters(self):
        """Initialize formatters"""
        return {
            'with_semicolon_at_end': self._format_with_semicolon_at_end,
            'before_semicolon': self._format_before_semicolon,
            'before_semicolon_with_if_for_pap': self._format_before_semicolon_with_if_for_pap,
            'empty_string_separated': self._format_empty_string_separated,
            'inside_condition': self._format_inside_condition
        }
    
    # ==================== Formatter Methods ====================
    
    def _format_with_semicolon_at_end(self, instruments, expression):
        """Formatter that adds semicolon at the end"""
        return ';'.join(instruments) + ';' if instruments else ''

    def _format_before_semicolon(self, instruments, expression):
        """Formatter that adds before semicolon"""
        return ';' + ';'.join(instruments) if instruments else ''

    def _format_before_semicolon_with_if_for_pap(self, instruments, expression):
        """Special formatter for PAP algorithm with if condition"""
        if not instruments:
            return ''
        return ';if(' + expression + ') ' + ';'.join(instruments)
    
    def _format_empty_string_separated(self, instruments, expression):
        """Formatter with empty string separation"""
        return ''.join(instruments)

    def _format_inside_condition(self, instruments, expression):
        """Formatter for inside condition"""
        return '||'.join(instruments) + '||' if instruments else ''
    
    # ==================== Position Calculation Methods ====================
    
    def _get_instrumentation_location_before(self, node):
        """Get instrumentation position (before node)"""
        if node is None:
            return None

        if node.type in [NodeType.ENTRYPOINT, NodeType.IFLOOP]:
            # should never get here
            assert False

        elif node.type in [NodeType.EXPRESSION, NodeType.IF, NodeType.RETURN, NodeType.CONTINUE, 
                          NodeType.BREAK, NodeType.THROW, NodeType.VARIABLE, NodeType.ASSEMBLY]:
            lines = node.source_mapping['lines']
            line = lines[0]
            col = node.source_mapping['starting_column']

        elif node.type in [NodeType.ENDIF, NodeType.ENDLOOP]:
            col = node.source_mapping['ending_column'] - 1
            lines = node.source_mapping['lines']
            line = lines[len(lines) - 1]

        elif node.type == NodeType.STARTLOOP:
            lines = node.source_mapping['lines']
            line = lines[0]
            col = node.source_mapping['starting_column']

        else:
            raise Exception('unknown scenario')

        return line, col

    def _get_instrumentation_location_after(self, node):
        """Get instrumentation position (after node)"""
        if node is None:
            return None

        if node.type in [NodeType.RETURN, NodeType.CONTINUE, NodeType.BREAK, NodeType.IF, 
                        NodeType.IFLOOP, NodeType.STARTLOOP]:
            # should never get here
            assert False

        elif node.type in [NodeType.EXPRESSION, NodeType.VARIABLE]:
            lines = node.source_mapping['lines']
            line = lines[0]
            col = node.source_mapping['ending_column']

        elif node.type == NodeType.ENTRYPOINT:
            lines = node.source_mapping['lines']
            line = lines[0]
            col = node.source_mapping['starting_column'] + 1

        elif node.type in [NodeType.ENDIF, NodeType.ENDLOOP, NodeType.ASSEMBLY]:
            lines = node.source_mapping['lines']
            line = lines[len(lines) - 1]
            col = node.source_mapping['ending_column']

        else:
            raise Exception('unknown scenario')

        return line, col
    
    # ==================== Instrumentation Position Calculation ====================
    
    def get_code_instrumentation_location(self, edge, graph, related_nodes_map, algorithm_type):
        """Get code instrumentation location (core method)"""
        from_node = edge.from_node
        to_node = edge.to_node

        instruments_formatter = None

        if to_node is exit_node:
            if from_node.type in [NodeType.RETURN, NodeType.THROW]:
                line, col = self._get_instrumentation_location_before(from_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif 'EXPRESSION revert(' in str(from_node):
                line, col = self._get_instrumentation_location_before(from_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif 'EXPRESSION selfdestruct(' in str(from_node):
                line, col = self._get_instrumentation_location_before(from_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif from_node.type in [NodeType.ENDIF, NodeType.ENDLOOP, NodeType.ENTRYPOINT, NodeType.ASSEMBLY]:
                line, col = self._get_instrumentation_location_after(from_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif from_node.type in [NodeType.EXPRESSION, NodeType.VARIABLE]:
                line, col = self._get_instrumentation_location_after(from_node)
                instruments_formatter = self.formatters['before_semicolon']
            else:
                raise Exception('unknown scenario')

        elif algorithm_type == "pap" and to_node.type == NodeType.IFLOOP:
            line, col = self._get_instrumentation_location_after(from_node)
            instruments_formatter = self.formatters['before_semicolon_with_if_for_pap']
            
        elif from_node is entry_node:
            if to_node.type in [NodeType.ENTRYPOINT]:
                line, col = self._get_instrumentation_location_after(to_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif to_node.type in [NodeType.ENDIF]:
                line, col = self._get_instrumentation_location_before(to_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif to_node.type in [NodeType.IF]:
                line, col = self._get_instrumentation_location_after(from_node)
                instruments_formatter = self.formatters['inside_condition']
            elif to_node.type in [NodeType.EXPRESSION, NodeType.VARIABLE]:
                line, col = self._get_instrumentation_location_before(to_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            elif to_node.type in [NodeType.IFLOOP, NodeType.STARTLOOP]:
                col, line, instruments_formatter = self._handle_loop_condition_detailed(edge, related_nodes_map)
            else:
                raise Exception('unknown scenario')

        elif self._is_loop_condition_node(edge, related_nodes_map):
            line, col = self._get_instrumentation_location_after(from_node)
            if algorithm_type == 'pap':
                instruments_formatter = self.formatters['before_semicolon_with_if_for_pap']
            else:
                instruments_formatter = self.formatters['before_semicolon']

        elif to_node.type == NodeType.IF:
            if from_node.type in [NodeType.EXPRESSION, NodeType.VARIABLE]:
                line, col = self._get_instrumentation_location_after(from_node)
                instruments_formatter = self.formatters['before_semicolon']
            else:
                line, col = self._get_instrumentation_location_before(to_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']

        elif to_node.type in [NodeType.ENDIF, NodeType.ENDLOOP]:
            line, col = self._get_instrumentation_location_before(to_node)
            instruments_formatter = self.formatters['with_semicolon_at_end']

        else:
            if from_node.type in [NodeType.IF, NodeType.ENDIF, NodeType.ENDLOOP]:
                line, col = self._get_instrumentation_location_before(to_node)
                instruments_formatter = self.formatters['with_semicolon_at_end']
            else:
                line, col = self._get_instrumentation_location_after(from_node)
                instruments_formatter = self.formatters['before_semicolon']

        return line, col, instruments_formatter

    def _handle_loop_condition_detailed(self, edge, related_nodes_map):
        """Detailed loop condition handling"""
        to_node = edge.to_node

        if to_node.type == NodeType.IFLOOP:
            start_loop_node = related_nodes_map.get(to_node, to_node)
        else:
            start_loop_node = to_node

        # scenario 1 - before first loop iteration
        if edge.reinit is None:
            line, col = self._get_instrumentation_location_before(start_loop_node)
            instruments_formatter = self.formatters['with_semicolon_at_end']
        # scenario 2 - each subsequent loop iteration
        else:
            # Find loop end node
            end_loop_node = related_nodes_map.get(start_loop_node, None)
            if end_loop_node:
                line, col = self._get_instrumentation_location_before(end_loop_node)
            else:
                line, col = self._get_instrumentation_location_before(start_loop_node)
            instruments_formatter = self.formatters['with_semicolon_at_end']
            
        return col, line, instruments_formatter

    def _is_loop_condition_node(self, edge, related_nodes_map):
        """Check if the target node is a loop start or condition node"""
        to_node = edge.to_node
        if to_node.type in [NodeType.STARTLOOP, NodeType.IFLOOP]:
            return True

        if to_node in related_nodes_map:
            related_node = related_nodes_map[to_node]
            if related_node.type in [NodeType.STARTLOOP, NodeType.IFLOOP]:
                return True

        return False

    def _get_if_loop_node(self, to_node, graph):
        """Get if-loop node"""
        children = get_children(to_node, graph)
        for child in children:
            if child.type == NodeType.IFLOOP:
                return child
        return None
    
    # ==================== Instrumentation Code Generation ====================
    
    def generate_epp_instrumentations(self, sol_instr, instrumented_edges, graph):
        """Generate instrumentation code for EPP algorithm"""
        related_nodes_map = self._build_related_nodes_map(graph)
        code_instrumentations = {}
        
        for edge in instrumented_edges:
            # Handle duplicate edge keys (resolve super inheritance)
            edge_key = get_unique_node_id(edge.from_node) + '-' + get_unique_node_id(edge.to_node)
            repeat = 0
            if edge_key in code_instrumentations:
                from_node_full_id = get_unique_node_id(edge.from_node)  
                to_node_full_id = get_unique_node_id(edge.to_node)
                repeat = repeat + 1
                edge_key = hash(str(repeat) + '|' + from_node_full_id + '|' + to_node_full_id)
                # This happens when using super inheritance
                print('warning: duplicate edge detected:', edge)

            instr_str = None
            contract = None
            node = None

            if edge.from_node is entry_node:
                # Entry node handling
                assert edge.init is not None and edge.inc is None and edge._reinit is None
                if edge.init != 0:
                    instr_str = [sol_instr.instr.get_val_init(edge.init)]
                    contract = edge.to_node.function.contract
                    node = edge.to_node

            elif edge.to_node is exit_node:
                # Exit node handling
                assert edge.init is None and edge.inc is not None and edge._reinit is None
                instr_str = [sol_instr.instr.get_epp_counter_and_flush(edge.inc)]
                contract = edge.from_node.function.contract
                node = edge.from_node

            else:
                # Regular edge handling
                assert edge.init is None and edge.inc is not None
                
                if edge._reinit is not None:
                    instr_path_counterinc_str = sol_instr.instr.get_epp_counter_and_flush(edge.inc)
                    instr_path_counter_reset_str = sol_instr.instr.get_epp_reset_counter(edge._reinit)
                    instr_str = [instr_path_counterinc_str, instr_path_counter_reset_str]
                    contract = edge.from_node.function.contract
                    node = edge.from_node
                else:
                    if edge.inc != 0:
                        instr_str = [sol_instr.instr.get_epp_counter(edge.inc)]
                        contract = edge.from_node.function.contract
                        node = edge.from_node

            if instr_str and len(instr_str) > 0:
                line, col, instruments_formatter = self.get_code_instrumentation_location(
                    edge, graph, related_nodes_map, 'epp')

                formatted_instr_str = self._format_instruction_string(
                    instr_str, instruments_formatter, expression=None)

                code_instrumentations[edge_key] = {
                    'instrumentation': formatted_instr_str,
                    'line': line,
                    'col': col,
                    'contract': contract,
                    'node': node,
                }

        return code_instrumentations

    def generate_pap_instrumentations(self, sol_instr, instrumented_edges, graph, slither):
        """Generate instrumentation code for PAP algorithm"""
        related_nodes_map = self._build_related_nodes_map_for_pap(graph)
        code_instrumentations = {}
        
        for edge in instrumented_edges:
            # Handle duplicate edge keys (resolve super inheritance)
            edge_key = get_unique_node_id(edge.from_node) + '-' + get_unique_node_id(edge.to_node)
            repeat = 0
            if edge_key in code_instrumentations:
                from_node_full_id = get_unique_node_id(edge.from_node)
                to_node_full_id = get_unique_node_id(edge.to_node)
                repeat = repeat + 1
                edge_key = hash(str(repeat) + '|' + from_node_full_id + '|' + to_node_full_id)
                print('warning: duplicate edge detected:', edge)

            instr_str = None
            contract = None
            node = None

            if edge.to_node is exit_node:
                # Exit node handling
                instr_str = [sol_instr.instr.get_pap_val(edge.inc)]
                contract = edge.from_node.function.contract
                node = edge.from_node
            else:
                # Regular edge handling
                if edge._reinit > 1:
                    instr_str = [sol_instr.instr.get_pap_counter(edge._reinit, edge.inc)]
                    contract = edge.from_node.function.contract
                    node = edge.from_node

            if instr_str and len(instr_str) > 0:
                line, col, instruments_formatter = self.get_code_instrumentation_location(
                    edge, graph, related_nodes_map, 'pap')
                
                expression = None
                if edge.to_node.type == NodeType.IFLOOP:
                    edge_file_name = edge.to_node.source_mapping['filename_used']
                    source_code = slither.crytic_compile.src_content[edge_file_name]
                    expression = source_code[int(edge.to_node.source_mapping['start']): 
                                           int(edge.to_node.source_mapping['start']) + 
                                           int(edge.to_node.source_mapping['length'])]

                formatted_instr_str = self._format_instruction_string(
                    instr_str, instruments_formatter, expression)

                code_instrumentations[edge_key] = {
                    'instrumentation': formatted_instr_str,
                    'line': line,
                    'col': col,
                    'contract': contract,
                    'node': node,
                }

        return code_instrumentations

    def _build_related_nodes_map(self, graph):
        """Build related nodes mapping (EPP algorithm)"""
        related_nodes_map = {}
        for vertex in graph.vertices:
            if vertex.type == NodeType.STARTLOOP:
                ifloop_node = self._get_if_loop_node(vertex, graph)
                if ifloop_node is not None:
                    related_nodes_map[ifloop_node] = vertex
        return related_nodes_map

    def _build_related_nodes_map_for_pap(self, graph):
        """Build related nodes mapping (PAP algorithm)"""
        related_nodes_map = {}
        for vertex in graph.vertices:
            if vertex.type == NodeType.STARTLOOP:
                ifloop_node = self._get_if_loop_node(vertex, graph)
                if ifloop_node is not None:
                    related_nodes_map[ifloop_node] = vertex
        return related_nodes_map

    def _format_instruction_string(self, instr_str, instruments_formatter, expression):
        """Format instrumentation string"""
        return instruments_formatter(instr_str, expression)
    
    # ==================== Auxiliary Functions ====================

    def aggregate_instrumentations(self, code_instrumentation):
        """
        Aggregate instrumentation code (simplified version)
        
        Group instrumentation instructions by contract and position. In inline assembly mode,
        all instructions are memory operations and execution order is not sensitive, 
        so no need to sort by type.
        
        Args:
            code_instrumentation: List of instrumentation instructions, each containing line, col, contract etc.
            
        Returns:
            list: Aggregated instrumentation instructions list, sorted by source code position
        """
        # Group by contract+line+column combination key
        instr_groups = {}
        for instr in code_instrumentation:
            contract = instr['contract']
            line = instr['line']
            col = instr['col']

            # Create unique key: contract_line_column
            key = f"{contract}_{line}_{col}"
            
            if key not in instr_groups:
                instr_groups[key] = []
            instr_groups[key].append(instr)

        # Build aggregated instrumentation instructions list
        result = []
        for instr_group in instr_groups.values():
            # Collect all instrumentation code
            aggregated_instructions = [instr['instrumentation'] for instr in instr_group]

            # Create aggregated instrumentation instruction
            aggregated_item = {
                'instr_group_line': instr_group[0]['line'],
                'instr_group_col': instr_group[0]['col'],
                'instr_group_contract': instr_group[0]['contract'],
                'aggregated_instrumentation': '\n'.join(aggregated_instructions)
            }
            result.append(aggregated_item)

        # Sort by position: first by line number, then by column number
        result.sort(key=lambda item: (item['instr_group_line'], item['instr_group_col']), reverse=False)
        
        return result
    
    # ==================== Source Code Injection ====================
    
    def inject_instrumentations_to_source(self, aggregated_instructions, slither):
        """Inject instrumentation code into source code"""
        contract_sources = {}
        
        for contract in slither.contracts:
            contract_file_name = contract.source_mapping['filename_used']
            original_source_code = slither.crytic_compile.src_content[contract_file_name]

            # Get instrumentation instructions for current contract
            contract_instructions = [
                instr for instr in aggregated_instructions 
                if instr['instr_group_contract'] is contract
            ]

            # Initialize source code injection state
            current_line = 1
            current_col = 1
            source_pos = 0
            modified_source = ""
            instr_index = 0

            # Get current instruction
            current_instruction = None
            if instr_index < len(contract_instructions):
                current_instruction = contract_instructions[instr_index]

            # Process source code character by character
            while source_pos < len(original_source_code):
                # Check if instruction needs to be inserted at current position
                if (current_instruction is not None and 
                    current_line == current_instruction['instr_group_line'] and 
                    current_col == current_instruction['instr_group_col']):
                    
                    # Insert aggregated instrumentation code
                    modified_source += current_instruction['aggregated_instrumentation']

                    # Move to next instruction
                    instr_index += 1
                    if instr_index < len(contract_instructions):
                        current_instruction = contract_instructions[instr_index]
                    else:
                        current_instruction = None

                # Copy original character
                char = original_source_code[source_pos]
                modified_source += char
                
                # Update position counters
                if char == '\n':
                    current_line += 1
                    current_col = 1
                else:
                    current_col += 1
                source_pos += 1

            contract_sources[contract] = modified_source

        return contract_sources
    
    # ==================== File Writing ====================
    
    def write_instrumented_contracts(self, contract_sources, output_directory):
        """Write instrumented contract files"""
        for contract, source_code in contract_sources.items():
            file_path = output_directory + contract.name + '.sol'
            write_to_file(file_path, 'w', source_code)


# ==================== Convenient Factory Functions ====================

def create_instrument():
    """
    Create code instrumenter
    
    Uses inline assembly mode with the following advantages:
    - No external contract dependencies
    - Directly generates Solidity inline assembly code
    - Better performance and lower gas consumption
    
    Returns:
        CodeInstrumenter: Instrumenter instance
    """
    return CodeInstrumenter()
