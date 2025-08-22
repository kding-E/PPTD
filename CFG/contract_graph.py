# Slither framework unified import  
from slither_imports import NodeType, SlitherNode
from utils import get_unique_node_id, get_node_function, get_non_ssa_node

# Entry and exit node constants
ENTRY_NODE_ID = -1
EXIT_NODE_ID = -2

entry_node = SlitherNode(NodeType.PLACEHOLDER, ENTRY_NODE_ID)
exit_node = SlitherNode(NodeType.PLACEHOLDER, EXIT_NODE_ID)


class ContractGraph:
    """
    Integrated contract graph class, containing Node and Edge as inner classes
    """
    
    class Node:
        """Node class as inner class of ContractGraph"""

        def __init__(self, node):
            self.node = node
            self.node_id = node.node_id
            self.in_edges = []
            self.out_edges = []

        def __str__(self):
            if self.node.node_id < 0:
                return str(self.node.node_id)
            else:
                return str(get_unique_node_id(self.node))

    class Edge:
        """Edge class as inner class of ContractGraph"""
        _edge_counter = [0]

        def __init__(self, from_node, to_node):
            self.edge_id = self._edge_counter[0]
            self._edge_counter[0] += 1
            
            self.from_node = from_node
            self.to_node = to_node
            
            # Edge mapping attributes
            self.back_edge_entry_mapping = None
            self.back_edge_exit_mapping = None
            
            # Control flow attributes
            self.true_son = False
            self.false_son = False
            self.is_call_site = False
            self.related_to_call_site_edge_id = None
            
            # Instrumentation attributes
            self.init = None
            self.inc = None
            self.re_init = None

        def is_instrumented(self):
            """Check if edge is instrumented"""
            return any([self.init is not None, 
                       self.inc is not None, 
                       self.re_init is not None])

        def __str__(self):
            from_id = (self.from_node.node_id if self.from_node.node_id < 0 
                      else get_unique_node_id(self.from_node))
            to_id = (self.to_node.node_id if self.to_node.node_id < 0 
                    else get_unique_node_id(self.to_node))
            return f'{from_id} -> {to_id}'

    def __init__(self, vertices, edges):
        """Initialize contract graph
        
        Args:
            vertices: Vertex set
            edges: Edge set
        """
        self.vertices = vertices
        self.edges = edges

    def group_by_contract_function(self):
        """Group graph by contract function"""
        contract_graphs = {}

        # Group vertices by function
        for vertex in self.vertices:
            function = get_node_function(vertex)
            contract = function.contract

            if contract not in contract_graphs:
                contract_graphs[contract] = {}

            if function not in contract_graphs[contract]:
                contract_graphs[contract][function] = {
                    'vertices': set(),
                    'edges': set()
                }

            contract_graphs[contract][function]['vertices'].add(vertex)

        # Group edges by function
        for edge in self.edges:
            function = get_node_function(edge.from_node)
            contract = function.contract

            if contract not in contract_graphs or function not in contract_graphs[contract]:
                raise ValueError('Edge node contains function not in vertices')

            contract_graphs[contract][function]['edges'].add(edge)

        # Create ContractGraph instances for each function
        for contract in contract_graphs:
            for function in contract_graphs[contract]:
                graph_data = contract_graphs[contract][function]
                contract_graphs[contract][function] = ContractGraph(
                    graph_data['vertices'], graph_data['edges'])

        return contract_graphs

    def create_graph_with_entry_exit_nodes(self):
        """Create new graph connected with entry and exit nodes"""
        roots_leaves_orphans = self._get_roots_leaves_orphans()

        # Copy original edges and add new vertices
        new_edges = set(self.edges)
        new_vertices = {entry_node, exit_node} | set(self.vertices)

        # Connect entry node to root nodes and orphan nodes
        root_and_orphan_nodes = roots_leaves_orphans['roots'].union(
            roots_leaves_orphans['orphans'])
        for vertex in root_and_orphan_nodes:
            new_edges.add(self.Edge(entry_node, vertex))

        # Connect leaf nodes and orphan nodes to exit node
        leaf_and_orphan_nodes = roots_leaves_orphans['leaves'].union(
            roots_leaves_orphans['orphans'])
        for vertex in leaf_and_orphan_nodes:
            new_edges.add(self.Edge(vertex, exit_node))

        return ContractGraphWithExitEntry(new_vertices, new_edges)

    def _get_nodes_without_incoming_edges(self):
        """Get nodes without incoming edges (orphan nodes and root nodes)"""
        nodes_with_incoming = {edge.to_node for edge in self.edges}
        return set(self.vertices) - nodes_with_incoming

    def _get_nodes_without_outgoing_edges(self):
        """Get nodes without outgoing edges (orphan nodes and leaf nodes)"""
        nodes_with_outgoing = {edge.from_node for edge in self.edges}
        return set(self.vertices) - nodes_with_outgoing

    def _get_roots_leaves_orphans(self):
        """Get root nodes, leaf nodes and orphan nodes"""
        nodes_without_incoming = self._get_nodes_without_incoming_edges()
        nodes_without_outgoing = self._get_nodes_without_outgoing_edges()

        orphans = nodes_without_incoming.intersection(nodes_without_outgoing)
        roots = nodes_without_incoming - orphans
        leaves = nodes_without_outgoing - orphans

        return {'roots': roots, 'leaves': leaves, 'orphans': orphans}

    @staticmethod
    def find_reachable_nodes(start_node, edges, visited_nodes):
        """Recursively find all nodes reachable from the starting node"""
        visited_nodes.add(start_node)
        for edge in edges:
            if edge.from_node is start_node and edge.to_node not in visited_nodes:
                ContractGraph.find_reachable_nodes(edge.to_node, edges, visited_nodes)

class ContractGraphWithExitEntry(ContractGraph):
    """Contract graph class with entry and exit nodes"""
    
    def __init__(self, vertices, edges):
        super().__init__(vertices, edges)
        if entry_node not in self.vertices or exit_node not in self.vertices:
            raise ValueError('Graph does not contain entry/exit nodes')

    def get_function_root_leaf_mapping(self):
        """Get mapping from functions to root nodes and leaf nodes"""
        root_to_leaves_mapping = self._find_root_to_leaves_paths()

        unreachable_code_info = []
        function_mapping = {}
        
        for root, leaves in root_to_leaves_mapping:
            non_ssa_root = get_non_ssa_node(root)

            # A function may have multiple root nodes, node_id of 0 is the real entry
            # Other root nodes may be unreachable dead code
            if non_ssa_root.node_id != 0:
                info = f"{get_unique_node_id(root)} lines:{non_ssa_root.source_mapping['lines']}"
                unreachable_code_info.append(info)

            function = get_node_function(root)
            if function not in function_mapping:
                function_mapping[function] = {'roots': set(), 'leaves': set()}

            function_mapping[function]['roots'].add(root)
            function_mapping[function]['leaves'].update(leaves)

        if unreachable_code_info:
            print('Unreachable code located at: ' + '\n'.join(unreachable_code_info))
        
        return function_mapping

    def group_by_contract_function(self):
        """Group graph by contract function, overriding parent method to handle entry/exit nodes"""
        contract_graphs = {}

        # Handle vertices (skip entry and exit nodes)
        for vertex in self.vertices:
            if vertex in (entry_node, exit_node):
                continue

            function = get_node_function(vertex)
            contract = function.contract

            if contract not in contract_graphs:
                contract_graphs[contract] = {}

            if function not in contract_graphs[contract]:
                contract_graphs[contract][function] = {
                    'vertices': set(),
                    'edges': set()
                }

            contract_graphs[contract][function]['vertices'].add(vertex)

        # Handle edges (skip edges starting from entry node)
        for edge in self.edges:
            assert edge.from_node is not exit_node, "There should be no edges starting from exit node"

            if edge.from_node is entry_node:
                continue

            function = get_node_function(edge.from_node)
            contract = function.contract

            if (contract not in contract_graphs or 
                function not in contract_graphs[contract]):
                raise ValueError('Edge node contains function not in vertices')

            contract_graphs[contract][function]['edges'].add(edge)

        # Add exit node to functions containing exit node
        for contract in contract_graphs:
            for function in contract_graphs[contract]:
                graph_data = contract_graphs[contract][function]

                # Check if there are edges pointing to exit node
                has_exit_edge = any(edge.to_node is exit_node 
                                  for edge in graph_data['edges'])

                if has_exit_edge:
                    graph_data['vertices'].add(exit_node)

                contract_graphs[contract][function] = ContractGraph(
                    graph_data['vertices'], graph_data['edges'])

        return contract_graphs

    def _find_root_to_leaves_paths(self):
        """Find all paths from root nodes to leaf nodes"""
        if not self.vertices:
            return []

        root_nodes = self._get_root_nodes()
        root_to_leaves = []

        for root in root_nodes:
            leaves = set()
            visited = set()
            self._find_leaves_from_node(root, leaves, visited)
            root_to_leaves.append((root, leaves))

        return root_to_leaves

    def _find_leaves_from_node(self, node, leaves, visited):
        """Recursively find leaf nodes from specified node"""
        if node in visited:
            return

        visited.add(node)

        if node is exit_node:
            return

        # Find edges starting from current node
        node_edges = [edge for edge in self.edges if edge.from_node is node]
        
        for edge in node_edges:
            if edge.from_node is not entry_node and edge.to_node is exit_node:
                leaves.add(node)
            else:
                self._find_leaves_from_node(edge.to_node, leaves, visited)

    def _get_root_nodes(self):
        """Get root nodes (nodes directly reachable from entry node)"""
        if not self.vertices:
            return set()

        return {edge.to_node for edge in self.edges 
                if edge.from_node is entry_node and edge.to_node is not exit_node}

    def _get_leaf_nodes(self):
        """Get leaf nodes (nodes directly connected to exit node)"""
        if not self.vertices:
            return set()

        return {edge.from_node for edge in self.edges 
                if edge.from_node is not entry_node and edge.to_node is exit_node}

    def get_roots_leaves_orphans(self):
        """Get root nodes, leaf nodes and orphan nodes (retained for compatibility)"""
        root_nodes = self._get_root_nodes()
        leaf_nodes = self._get_leaf_nodes()

        # Orphan nodes are both root nodes and leaf nodes
        orphan_nodes = root_nodes.intersection(leaf_nodes)
        actual_roots = root_nodes - orphan_nodes
        actual_leaves = leaf_nodes - orphan_nodes

        return {
            'roots': actual_roots,
            'leaves': actual_leaves,
            'orphans': orphan_nodes
        }


# Backward compatibility aliases
Graph = ContractGraph
GraphWithExitEntry = ContractGraphWithExitEntry
Node = ContractGraph.Node
Edge = ContractGraph.Edge
