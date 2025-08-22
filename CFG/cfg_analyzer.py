"""Control Flow Graph Analyzer Module"""

from slither_imports import (
    Node, NodeType, SolidityCall, Operation, Index, OperationWithLValue, 
    PhiCallback, Phi, Member, InternalCall, HighLevelCall, 
    InternalDynamicCall, LowLevelCall, NewContract, StateVariable,
    ReferenceVariable, LocalIRVariable, StateIRVariable, slithirConverter
)

from CFG.contract_graph import entry_node, exit_node, Edge, ContractGraph as Graph
from utils import (
    full_function_signature, compose_full_function_signature
)


class CyclicReferenceException(Exception):
    """Cyclic reference exception"""
    pass


class CFGAnalyzer:
    """Control Flow Graph Analyzer"""
    
    def __init__(self):
        """Initialize Control Flow Graph Analyzer"""
        pass
    
    def _build_ssa_connections(self, node, last_instruction, edges):
        """Build connections between SSA nodes"""
        if node.type == NodeType.IF:
            # Handle true branch of conditional node
            true_node = node.son_true
            if true_node:
                first_instruction = self._get_first_instruction(true_node)
                edge = Edge(last_instruction, first_instruction)
                edge._true_son = True
                edges.add(edge)

            # Handle false branch of conditional node
            false_node = node.son_false
            if false_node:
                first_instruction = self._get_first_instruction(false_node)
                edge = Edge(last_instruction, first_instruction)
                edge._false_son = True
                edges.add(edge)
        elif node.sons:
            # Handle all child nodes of other types
            for son in node.sons:
                first_instruction = self._get_first_instruction(son)
                edge = Edge(last_instruction, first_instruction)
                edges.add(edge)
    
    def _get_first_instruction(self, node):
        """Get the first instruction of a node"""
        if node.irs_ssa is None or len(node.irs_ssa) == 0:
            return node
        else:
            return node.irs_ssa[0]

    def _get_last_instruction(self, node):
        """Get the last instruction of a node"""
        if len(node.irs_ssa) > 0:
            return node.irs_ssa[len(node.irs_ssa) - 1]
        else:
            return node

    def get_ssa_position(self, ssa_node):
        """Get the position of an SSA node within its parent node"""
    def get_ssa_position(self, ssa_node):
        """Get the position of an SSA node within its parent node"""
        if not isinstance(ssa_node, Operation):
            return 0
        
        node = ssa_node.node
        if node.irs_ssa is None:
            return 0
        
        for i, ir_ssa in enumerate(node.irs_ssa):
            if ir_ssa is ssa_node:
                return i
        
        return 0

    def get_non_ssa_node(self, node):
        """Get the corresponding non-SSA node of an SSA node"""
        if isinstance(node, Operation):
            return node.node
        else:
            return node

    def get_last_ssa_in_node(self, first_ssa, ssa_graph):
        """Get the last SSA instruction in a node"""
        if self.get_ssa_position(first_ssa) != 0:
            raise ValueError("The input is not the first SSA instruction of the node")

        last_ssa = first_ssa
        next_ssa = self._get_next_ssa_in_node(first_ssa, ssa_graph)
        
        while next_ssa is not None:
            last_ssa = next_ssa
            next_ssa = self._get_next_ssa_in_node(next_ssa, ssa_graph)

        return last_ssa if last_ssa is not None else first_ssa

    def _get_next_ssa_in_node(self, ssa_node, ssa_graph):
        """Get the next SSA instruction within the same node"""
        for edge in ssa_graph.edges:
            if edge.from_node is ssa_node:
                non_ssa_from = self.get_non_ssa_node(edge.from_node)
                non_ssa_to = self.get_non_ssa_node(edge.to_node)
                
                # If it's an edge within the same node, return the target node
                if non_ssa_from is non_ssa_to:
                    return edge.to_node
        
        return None

    def get_main_ssa_root(self, ssa_roots):
        """Get the main root node from multiple SSA root nodes"""
        if len(ssa_roots) == 0:
            raise ValueError("SSA root node list cannot be empty")

        if len(ssa_roots) == 1:
            return ssa_roots[0]

        # Prioritize non-Operation type root nodes
        for root in ssa_roots:
            if not isinstance(root, Operation):
                return root

        # If all are Operation types, select the node with node_id of 0
        for ssa_root in ssa_roots:
            if isinstance(ssa_root, Operation) and ssa_root.node.node_id == 0:
                return ssa_root

        return ssa_roots[0]

    def get_node_ssa_call_sites(self, ssa_node, ssa_graph, function_mapping, processed_nodes):
        """Get SSA call site information within a node"""
        call_site_roots_and_leaves = []

        if not isinstance(ssa_node, Operation):
            return call_site_roots_and_leaves

        if self.get_ssa_position(ssa_node) != 0:
            raise ValueError("The input is not the first SSA instruction of the node")

        # Get call site information for the current SSA node
        self._get_call_site_roots_and_leaves(
            ssa_node, function_mapping, call_site_roots_and_leaves
        )

        # Process other SSA instructions within the same node
        next_ssa = self._get_next_ssa_in_node(ssa_node, ssa_graph)
        while next_ssa is not None:
            if next_ssa in processed_nodes:
                raise ValueError("SSA node has already been processed")
            
            processed_nodes.add(next_ssa)
            self._get_call_site_roots_and_leaves(
                next_ssa, function_mapping, call_site_roots_and_leaves
            )
            next_ssa = self._get_next_ssa_in_node(next_ssa, ssa_graph)

        return call_site_roots_and_leaves

    def _get_call_site_roots_and_leaves(self, ssa_node, function_mapping, call_site_roots_and_leaves):
        """Get root and leaf node information of call sites"""
        if not isinstance(ssa_node, Operation):
            raise ValueError("The input is not of Operation type")

        if isinstance(ssa_node, SolidityCall):
            # Skip processing Solidity calls
            pass
        else:
            # Use existing _extract_call_site_info method
            self._extract_call_site_info(
                ssa_node, function_mapping, call_site_roots_and_leaves
            )

    def add_edge_with_cycle_check(self, edge, edge_set):
        """Add edge to the set while checking for cyclic references"""
        for existing_edge in edge_set:
            if (existing_edge.from_node is edge.from_node and 
                existing_edge.to_node is edge.to_node and 
                edge._true_son == existing_edge._true_son and 
                edge._false_son == existing_edge._false_son):
                raise CyclicReferenceException('Cyclic reference detected')

        edge_set.add(edge)

    def build_consolidated_graph(self, function_ssa_graph, ssa_node, consolidated_vertices, 
                                consolidated_edges, processed_ssa_nodes, function_mapping):
        """Recursively build consolidated graph"""
        # Only process the first SSA instruction in a node
        if self.get_ssa_position(ssa_node) != 0:
            raise ValueError("Should only process the first SSA instruction of a node")

        if ssa_node in processed_ssa_nodes:
            return

        processed_ssa_nodes.add(ssa_node)

        # Skip entry/exit nodes
        if ssa_node is exit_node:
            return

        non_ssa_node = self.get_non_ssa_node(ssa_node)
        if non_ssa_node in consolidated_vertices:
            raise ValueError("Non-SSA node already exists in consolidated vertices")
        
        consolidated_vertices.add(non_ssa_node)

        # Process call site chains within the node
        node_ssa_call_sites = self.get_node_ssa_call_sites(
            ssa_node, function_ssa_graph, function_mapping, processed_ssa_nodes
        )

        outer_most_call_site_info = None
        call_entry_edge = None

        if len(node_ssa_call_sites) > 0:
            # Process the innermost call site
            inner_most_call_site_info = node_ssa_call_sites[0]
            inner_most_ssa_roots = inner_most_call_site_info['roots']

            inner_most_main_ssa_root = self.get_main_ssa_root(inner_most_ssa_roots)
            non_ssa_inner_most_root_node = self.get_non_ssa_node(inner_most_main_ssa_root)
            
            if non_ssa_node is non_ssa_inner_most_root_node:
                raise ValueError("Caller and callee cannot be the same node")

            call_entry_edge = Edge(non_ssa_node, non_ssa_inner_most_root_node)
            call_entry_edge._is_call_site = True
            self.add_edge_with_cycle_check(call_entry_edge, consolidated_edges)

            if len(node_ssa_call_sites) == 1:
                outer_most_call_site_info = inner_most_call_site_info
            else:
                # Process chained calls with multiple call sites
                outer_most_call_site_info = node_ssa_call_sites[-1]
                self._process_chained_call_sites(
                    node_ssa_call_sites, consolidated_edges
                )

        # Process outgoing edges from the node
        self._process_node_outgoing_edges(
            ssa_node, function_ssa_graph, outer_most_call_site_info, 
            call_entry_edge, consolidated_vertices, consolidated_edges, 
            processed_ssa_nodes, function_mapping
        )

    def _process_chained_call_sites(self, node_ssa_call_sites, consolidated_edges):
        """Process chained call sites"""
        current_call_site_info = node_ssa_call_sites[0]
        
        for i in range(1, len(node_ssa_call_sites)):
            current_ssa_leaves = current_call_site_info['leaves']
            next_call_site_info = node_ssa_call_sites[i]
            next_ssa_roots = next_call_site_info['roots']
            next_ssa_main_root = self.get_main_ssa_root(next_ssa_roots)
            next_non_ssa_main_root = self.get_non_ssa_node(next_ssa_main_root)

            for ssa_leaf in current_ssa_leaves:
                non_ssa_leaf = self.get_non_ssa_node(ssa_leaf)
                if non_ssa_leaf is next_non_ssa_main_root:
                    raise ValueError("Cycle detected in call chain")

                chain_edge = Edge(non_ssa_leaf, next_non_ssa_main_root)
                self.add_edge_with_cycle_check(chain_edge, consolidated_edges)

            current_call_site_info = next_call_site_info

    def _process_node_outgoing_edges(self, ssa_node, function_ssa_graph, outer_most_call_site_info, 
                                    call_entry_edge, consolidated_vertices, consolidated_edges, 
                                    processed_ssa_nodes, function_mapping):
        """Process outgoing edges from a node"""
        last_ssa_in_node = self.get_last_ssa_in_node(ssa_node, function_ssa_graph)

        outgoing_edges = []
        for edge in function_ssa_graph.edges:
            if (edge.from_node is last_ssa_in_node and 
                edge.to_node is not exit_node and 
                self.get_ssa_position(edge.to_node) == 0):
                
                non_ssa_from_node = self.get_non_ssa_node(edge.from_node)
                non_ssa_to_node = self.get_non_ssa_node(edge.to_node)
                
                if non_ssa_from_node is non_ssa_to_node:
                    raise ValueError("Source and target nodes of outgoing edge cannot be the same")
                    
                outgoing_edges.append(edge)

        # Create actual outgoing edges
        for edge in outgoing_edges:
            non_ssa_from_node = self.get_non_ssa_node(edge.from_node)
            non_ssa_to_node = self.get_non_ssa_node(edge.to_node)

            if outer_most_call_site_info is not None:
                # Case with call site: connect from call leaf nodes to target node
                outer_ssa_leaves = outer_most_call_site_info['leaves']
                for leaf in outer_ssa_leaves:
                    non_ssa_leaf = self.get_non_ssa_node(leaf)
                    if non_ssa_leaf is non_ssa_to_node:
                        raise ValueError("Call leaf node cannot be the same as target node")

                    exit_edge = Edge(non_ssa_leaf, non_ssa_to_node)
                    exit_edge._true_son = edge._true_son
                    exit_edge._false_son = edge._false_son
                    exit_edge._related_to_call_site_edge_id = call_entry_edge.edge_id
                    self.add_edge_with_cycle_check(exit_edge, consolidated_edges)
            else:
                # Case without call site: direct connection
                direct_edge = Edge(non_ssa_from_node, non_ssa_to_node)
                direct_edge._true_son = edge._true_son
                direct_edge._false_son = edge._false_son
                self.add_edge_with_cycle_check(direct_edge, consolidated_edges)

            # Recursively process target node
            self.build_consolidated_graph(
                function_ssa_graph, edge.to_node, consolidated_vertices,
                consolidated_edges, processed_ssa_nodes, function_mapping
            )

    def build_function_graph(self, function):
        """Build control flow graph for a single function"""
        vertices = set()
        edges = set()

        for node in function.nodes:
            vertices.add(node)
            self._build_graph_edges(node, edges)

        return Graph(vertices, edges)
    
    def _build_graph_edges(self, node, edges):
        """Build graph edge connections"""
        if node.type == NodeType.IF:
            # Handle true branch of conditional node
            true_node = node.son_true
            if true_node:
                edge = Edge(node, true_node)
                edge._true_son = True
                edges.add(edge)

            # Handle false branch of conditional node
            false_node = node.son_false
            if false_node:
                edge = Edge(node, false_node)
                edge._true_son = False
                edges.add(edge)
        else:
            # Handle all child nodes of other types
            for son in node.sons:
                edges.add(Edge(node, son))
    
    def build_function_ssa_graph(self, function):
        """Build SSA control flow graph for a single function"""
        vertices = set()
        edges = set()
        
        for node in function.nodes:
            # Process nodes containing SSA instructions
            if len(node.irs_ssa) > 0:
                first_ir_ssa = node.irs_ssa[0]
                vertices.add(first_ir_ssa)

                # Create edges between SSA instructions within the same node
                current_ir_ssa = first_ir_ssa
                if len(node.irs_ssa) > 1:
                    for i in range(1, len(node.irs_ssa)):
                        next_ir_ssa = node.irs_ssa[i]
                        vertices.add(next_ir_ssa)
                        ssa_edge = Edge(current_ir_ssa, next_ir_ssa)
                        edges.add(ssa_edge)
                        current_ir_ssa = next_ir_ssa
            else:
                # Add nodes without SSA instructions directly
                vertices.add(node)

            # Get the last instruction of the node for connecting to other nodes
            last_instruction = self._get_last_instruction(node)
            
            # Skip revert and selfdestruct expressions
            if 'EXPRESSION revert' in str(node) or 'EXPRESSION selfdestruct' in str(node):
                continue

            # Create connections between nodes
            self._build_ssa_connections(node, last_instruction, edges)

        return Graph(vertices, edges)

    def build_contracts_graph(self, contracts):
        """Build control flow graph for all contracts"""
        vertices = set()
        edges = set()
        
        for contract in contracts:
            for function in contract.functions + contract.modifiers:
                function_graph = self.build_function_graph(function)
                vertices = vertices.union(function_graph.vertices)
                edges = edges.union(function_graph.edges)

        return Graph(vertices, edges)

    def build_contracts_ssa_graph(self, contracts):
        """Build SSA control flow graph for all contracts"""
        vertices = set()
        edges = set()

        for contract in contracts:
            for function in contract.functions + contract.modifiers:
                function_graph = self.build_function_ssa_graph(function)
                vertices |= function_graph.vertices
                edges |= function_graph.edges

        return Graph(vertices, edges)

    def build_consolidated_contracts_graph(self, contracts_ssa_graph_with_entry_exit):
        """Consolidate contract graph, handling function call relationships"""
        function_mapping = contracts_ssa_graph_with_entry_exit.get_function_root_leaf_mapping()
        contracts = contracts_ssa_graph_with_entry_exit.group_by_contract_function()

        vertices = set()
        edges = set()
        all_processed_ssa_nodes = set()

        # Process each function of each contract
        for contract in contracts:
            for function in contract.functions + contract.modifiers:
                function_key = function 
                if function_key not in function_mapping:
                    continue
                
                function_roots_leaves = function_mapping[function_key]
                roots = function_roots_leaves if isinstance(function_roots_leaves, list) else [function_roots_leaves[0]] if isinstance(function_roots_leaves, tuple) else function_roots_leaves.get('roots', [])
                
                if isinstance(roots, list) and len(roots) > 0:
                    main_ssa_root = self.get_main_ssa_root(roots)
                elif hasattr(function_roots_leaves, '__iter__') and not isinstance(function_roots_leaves, str):
                    main_ssa_root = self.get_main_ssa_root(list(function_roots_leaves))
                else:
                    continue

                # Build consolidated graph for the function
                function_ssa_graph = contracts_ssa_graph_with_entry_exit 
                consolidated_vertices = set()
                consolidated_edges = set()
                processed_ssa_nodes = set()
                
                try:
                    self.build_consolidated_graph(
                        function_ssa_graph, main_ssa_root, consolidated_vertices, 
                        consolidated_edges, processed_ssa_nodes, function_mapping
                    )
                    vertices |= consolidated_vertices
                    edges |= consolidated_edges
                    all_processed_ssa_nodes |= processed_ssa_nodes
                except Exception as e:
                    # If consolidation fails, use simplified processing
                    print(f"Error consolidating function {function.name}: {e}")
                    self._process_function_call_nodes_simple(
                        function, function_mapping, vertices, edges
                    )

        # Verify all SSA nodes have been processed
        for vertex in contracts_ssa_graph_with_entry_exit.vertices:
            if vertex is entry_node or vertex is exit_node:
                continue
            if isinstance(vertex, Operation) and vertex not in all_processed_ssa_nodes:
                print(f"Warning: SSA node {vertex} was not processed")

        # Create graph with entry/exit nodes
        merged_graph = Graph(vertices, edges)
        try:
            return merged_graph.create_graph_with_entry_exit_nodes()
        except:
            # If creating entry/exit nodes fails, return original graph
            return merged_graph
    
    def _process_function_call_nodes_simple(self, function, function_mapping, vertices, edges):
        """Simplified function call node processing"""
        vertices.add(function)
        
        if function not in function_mapping:
            return

        for node in function.nodes:
            vertices.add(node)
            for ir in node.irs:
                dest_function_signatures = []
                self._extract_call_site_info(
                    ir, function_mapping, dest_function_signatures
                )

                for dest_info in dest_function_signatures:
                    roots = dest_info['roots']
                    leaves = dest_info['leaves']
                    
                    # Add call edges
                    for root in roots:
                        call_edge = Edge(node, root)
                        edges.add(call_edge)

                    # Add return edges
                    for leaf in leaves:
                        return_edge = Edge(leaf, node)
                        edges.add(return_edge)
    
    def _process_function_call_nodes(self, node, function_mapping, edges):
        """Process function call nodes, add call and return edges"""
        for ir in node.irs:
            dest_function_signatures = []
            self._extract_call_site_info(
                ir, function_mapping, dest_function_signatures
            )

            for dest_info in dest_function_signatures:
                roots = dest_info['roots']
                leaves = dest_info['leaves']
                
                # Add call edges
                for root in roots:
                    call_edge = Edge(node, root)
                    edges.add(call_edge)

                # Add return edges
                for leaf in leaves:
                    return_edge = Edge(leaf, node)
                    edges.add(return_edge)

    def _find_constructor_mapping(self, function_mapping, contract_name, function_name, args):
        """Find root and leaf node mapping for constructor"""
        target_signature = compose_full_function_signature(
            contract_name, function_name, args
        )

        for function, root_and_leaves_nodes in function_mapping.items():
            function_signature = full_function_signature(function)
            if function_signature == target_signature:
                return root_and_leaves_nodes

        raise Exception(f'Constructor mapping not found: {target_signature}')

    def _extract_call_site_info(self, ir, function_mapping, dest_function_signatures):
        """Extract call site information"""
        if isinstance(ir, (InternalCall, InternalDynamicCall, HighLevelCall, SolidityCall)):
            self._handle_function_call(ir, function_mapping, dest_function_signatures)
        elif isinstance(ir, NewContract):
            self._handle_contract_creation(ir, function_mapping, dest_function_signatures)
        elif isinstance(ir, LowLevelCall):
            # Skip low-level calls
            pass

    def _handle_function_call(self, ir, function_mapping, dest_function_signatures):
        """Handle function call"""
        if ir.function not in function_mapping:
            # Skip processing abstract functions
            return
        
        root_and_leaves = function_mapping[ir.function]
        self._append_call_info(root_and_leaves, dest_function_signatures)

    def _handle_contract_creation(self, ir, function_mapping, dest_function_signatures):
        """Handle contract creation"""
        args = slithirConverter.convert_arguments(ir.arguments)
        args = [a[0] for a in args if a[0] != 'uint256']
        
        root_and_leaves = self._find_constructor_mapping(
            function_mapping,
            str(ir.contract_name),
            'constructor',
            args
        )
        self._append_call_info(root_and_leaves, dest_function_signatures)
    
    def _append_call_info(self, root_and_leaves, dest_function_signatures):
        """Append call information to target list"""
        roots = [root_and_leaves[0]] if not isinstance(root_and_leaves[0], (list, tuple)) else root_and_leaves[0]
        leaves = root_and_leaves[1] if isinstance(root_and_leaves[1], (list, tuple)) else [root_and_leaves[1]]
        
        dest_function_signatures.append({
            'roots': roots,
            'leaves': leaves
        })

    def _convert_ssa_variable(self, variable):
        """Convert SSA variable to non-SSA variable"""
        if isinstance(variable, StateIRVariable):
            contract = variable.contract
            non_ssa_var = contract.get_state_variable_from_name(variable.name)
            return non_ssa_var
        
        if isinstance(variable, LocalIRVariable):
            function = variable.function
            non_ssa_var = function.get_local_variable_from_name(variable.name)
            return non_ssa_var
        
        raise ValueError(f"Unsupported variable type: {type(variable)}")

    def convert_read_write_variables(self, ssa_vars_read, ssa_vars_written):
        """Convert read/write variables from SSA form to regular form"""
        # Deduplicate and convert read variables
        unique_ssa_vars_read = list(set(ssa_vars_read))
        vars_read = [self._convert_ssa_variable(var) for var in unique_ssa_vars_read]
        state_vars_read = [var for var in vars_read if isinstance(var, StateVariable)]
        
        # Deduplicate and convert written variables
        unique_ssa_vars_written = list(set(ssa_vars_written))
        vars_written = [self._convert_ssa_variable(var) for var in unique_ssa_vars_written]
        state_vars_written = [var for var in vars_written if isinstance(var, StateVariable)]
        
        return state_vars_read, state_vars_written

    def extract_ssa_read_write_variables(self, ir, ssa_vars_read, ssa_vars_written):
        """Extract SSA read/write variables"""
        # Handle read variables
        if not isinstance(ir, (Phi, Index, Member)):
            self._extract_read_variables(ir, ssa_vars_read)
        elif isinstance(ir, (Member, Index)):
            self._extract_member_index_variables(ir, ssa_vars_read)
        
        # Handle written variables
        if isinstance(ir, OperationWithLValue):
            self._extract_written_variable(ir, ssa_vars_written)

    def _extract_read_variables(self, ir, ssa_vars_read):
        """Extract read variables from regular operations"""
        for var in ir.read:
            if isinstance(var, (StateIRVariable, LocalIRVariable)):
                ssa_vars_read.append(var)
            elif isinstance(var, ReferenceVariable):
                origin = var.points_to_origin
                if isinstance(origin, (StateIRVariable, LocalIRVariable)):
                    ssa_vars_read.append(origin)

    def _extract_member_index_variables(self, ir, ssa_vars_read):
        """Extract variables from Member and Index operations"""
        if isinstance(ir.variable_right, (StateIRVariable, LocalIRVariable)):
            ssa_vars_read.append(ir.variable_right)
        elif isinstance(ir.variable_right, ReferenceVariable):
            origin = ir.variable_right.points_to_origin
            if isinstance(origin, (StateIRVariable, LocalIRVariable)):
                ssa_vars_read.append(origin)

    def _extract_written_variable(self, ir, ssa_vars_written):
        """Extract written variables"""
        var = ir.lvalue
        if isinstance(var, ReferenceVariable):
            var = var.points_to_origin
        
        if var and isinstance(var, (StateIRVariable, LocalIRVariable)):
            ssa_vars_written.append(var)

    def is_slithir_variable(self, ir):
        """Check if it is a SlithIR variable"""
        if isinstance(ir, PhiCallback):
            return False
        
        if isinstance(ir, OperationWithLValue):
            var = ir.lvalue
            if isinstance(var, ReferenceVariable):
                var = var.points_to_origin

            if var and isinstance(var, (StateIRVariable, LocalIRVariable)):
                if isinstance(ir, PhiCallback):
                    return False

        return True

    def extract_call_site_edges(self, contracts_graph_with_entry_exit):
        """Extract call site edge connection information"""
        function_mapping = contracts_graph_with_entry_exit.get_function_root_leaf_mapping()
        call_site_edges = set()

        for edge in contracts_graph_with_entry_exit.edges:
            if edge.from_node is entry_node and edge.to_node is exit_node:
                continue

            self._process_call_site_edge(
                edge, function_mapping, call_site_edges
            )

        return call_site_edges
    
    def _process_call_site_edge(self, edge, function_mapping, call_site_edges):
        """Process a single call site edge"""
        call_site_info_list = []

        if isinstance(edge.from_node, Operation) and not isinstance(edge.from_node, SolidityCall):
            self._extract_call_site_info(
                edge.from_node, function_mapping, call_site_info_list
            )
                
        for call_site_info in call_site_info_list:
            self._create_call_site_edges(edge, call_site_info, call_site_edges)
    
    def _create_call_site_edges(self, original_edge, call_site_info, call_site_edges):
        """Create entry and exit edges for call sites"""
        roots = call_site_info['roots']
        leaves = call_site_info['leaves']
        
        for root in roots:
            # Create call entry edge
            entry_edge = Edge(original_edge.from_node, root)
            entry_edge._is_call_site = True
            call_site_edges.add(entry_edge)

            # Create call exit edge
            for leaf in leaves:
                exit_edge = Edge(leaf, original_edge.from_node)
                exit_edge._related_to_call_site_edge_id = entry_edge.edge_id
                call_site_edges.add(exit_edge)
