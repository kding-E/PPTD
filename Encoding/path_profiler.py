"""
Path Profiler - Integrates EPP (Efficient Path Profiling) and PAP (Path-aware Profiling) algorithms
Eliminates code redundancy and provides unified path profiling interface
"""
import sys
from collections import defaultdict
from enum import Enum
from typing import List, Dict, Set, Tuple, Any, Optional

from CFG.contract_graph import Edge, Node
from utils import ( get_parents, get_exit, get_entry, 
    get_outgoing_edges, get_incoming_edges, get_children
)


class ProfilerType(Enum):
    """Path profiler type enumeration"""
    EPP = "efficient_path_profiling"  
    PAP = "profiling_all_paths"       


class PathProfiler:
    """
    Unified path profiler class
    Supports both EPP (Efficient Path Profiling) and PAP (Path-aware Profiling) algorithms
    """
    
    def __init__(self, profiler_type: ProfilerType) -> None:
        """
        Initialize path profiler
        
        Args:
            profiler_type: Profiler type (EPP or PAP)
        """
        self.profiler_type = profiler_type
    
    @staticmethod
    def _verify_graph_structure(vertices: Set, graph: List[Edge]) -> None:
        """Verify that the graph has unique entry and exit nodes"""
        entry_count = 0
        exit_count = 0

        for vertex in vertices:
            parents = get_parents(vertex, graph)
            if len(parents) == 0:
                entry_count += 1
            children = get_children(vertex, graph)
            if len(children) == 0:
                exit_count += 1

        if entry_count != 1 or exit_count != 1:
            raise Exception("CFG must have exactly one entry and one exit node")
    
    @staticmethod
    def _dfs_find_back_edges(current_node, graph: List[Edge], color: defaultdict, back_edges: List[Edge]) -> None:
        """Use DFS to find back edges in the graph"""
        color[current_node] = 1
        outgoing_edges = get_outgoing_edges(current_node, graph)
        for edge in outgoing_edges:
            target_node = edge.to_node
            if color[target_node] != 0:
                if color[target_node] == 1:  # Found back edge
                    back_edges.append(edge)
            else:
                PathProfiler._dfs_find_back_edges(target_node, graph, color, back_edges)
        color[current_node] = 2
    
    @staticmethod
    def _topological_sort(vertices: Set, graph: List[Edge]) -> List:
        """Topological sort of the graph"""
        incoming_edge_count = defaultdict(int)
        for target_node in vertices:
            incoming_edges = get_incoming_edges(target_node, graph)
            incoming_edge_count[target_node] = len(incoming_edges)

        sorted_nodes = []
        for node in vertices:
            if incoming_edge_count[node] == 0:
                sorted_nodes.append(node)

        index = 0
        while index < len(sorted_nodes):
            current_node = sorted_nodes[index]
            children = get_children(current_node, graph)
            for child in children:
                edge_count = incoming_edge_count[child] - 1
                incoming_edge_count[child] = edge_count
                if edge_count == 0:
                    sorted_nodes.append(child)
            index += 1

        return sorted_nodes
    
    @staticmethod
    def _path_exists_in_tree(source_node, target_node, tree: Set[Edge], visited: Set) -> bool:
        """Check if there is a path from source_node to target_node in the tree"""
        if source_node is target_node:
            return True

        visited.add(source_node)
        for edge in tree:
            next_node = None
            if edge.to_node is source_node:
                next_node = edge.from_node
            elif edge.from_node is source_node:
                next_node = edge.to_node

            if next_node is not None and next_node not in visited:
                if PathProfiler._path_exists_in_tree(next_node, target_node, tree, visited):
                    return True

        return False
    
    @staticmethod
    def _generate_maximum_spanning_tree(graph: List[Edge]) -> Set[Edge]:
        """Generate maximum spanning tree"""
        spanning_tree = set()

        for edge in graph:
            source_node = edge.from_node
            target_node = edge.to_node

            visited = set()
            if not PathProfiler._path_exists_in_tree(source_node, target_node, spanning_tree, visited):
                spanning_tree.add(edge)

        return spanning_tree
    
    @staticmethod
    def _calculate_increment(source_vertex, target_vertex, edge_values: Dict, tree: Set[Edge], visited: Set) -> Optional[int]:
        """
        Calculate increment value
        
        Args:
            source_vertex: Source vertex
            target_vertex: Target vertex
            edge_values: Edge values dictionary
            tree: Spanning tree
            visited: Set of visited nodes
            
        Returns:
            Optional[int]: Increment value, returns None if cannot be calculated
        """
        if source_vertex is target_vertex:
            return 0

        visited.add(source_vertex)
        for edge in tree:
            next_node = None
            if edge.to_node is source_vertex:
                next_node = edge.from_node
            elif edge.from_node is source_vertex:
                next_node = edge.to_node

            if next_node is not None and next_node not in visited:
                next_increment = PathProfiler._calculate_increment(next_node, target_vertex, edge_values, tree, visited)
                if next_increment is not None:  # Check if calculation was successful
                    edge_value = edge_values[edge]
                    if edge.to_node is source_vertex:
                        edge_value = -edge_value
                    return next_increment + edge_value

        return None  # Return None when no path can be found
    
    def _generate_epp_edge_values(self, sorted_nodes: List, graph: List[Edge]) -> Dict:
        """EPP Algorithm: Generate edge values"""
        edge_values = defaultdict(int)
        path_counts = defaultdict(int)

        for node in reversed(sorted_nodes):
            outgoing_edges = get_outgoing_edges(node, graph)
            if len(outgoing_edges) == 0:
                path_counts[node] = 1
            else:
                path_counts[node] = 0
                for edge in outgoing_edges:
                    edge_values[edge] = path_counts[node]
                    target_block = edge.to_node
                    path_counts[node] += path_counts[target_block]

        return edge_values
    
    def _generate_pap_edge_values(self, vertices_with_in_out: List[Node]) -> Tuple[Dict, Dict]:
        """PAP Algorithm: Generate s and i values"""
        s_values = defaultdict(int)
        i_values = defaultdict(int)

        for vertex in vertices_with_in_out:
            if vertex.node_in > 1:
                edge_counter = 0
                for edge in vertex.in_edges:
                    s_values[edge] = vertex.node_in
                    i_values[edge] = edge_counter
                    edge_counter += 1

        return s_values, i_values
    
    def _generate_edge_increments(self, graph: List[Edge], edge_values: Dict, spanning_tree: Set[Edge]) -> Dict:
        """
        Generate increment dictionary
        
        Args:
            graph: List of graph edges
            edge_values: Edge values dictionary
            spanning_tree: Spanning tree
            
        Returns:
            Dict: Increment dictionary
            
        Raises:
            Exception: When unable to calculate chord edge increment
        """
        increments = defaultdict(int)

        for edge in graph:
            if edge not in spanning_tree:
                source_node = edge.from_node
                target_node = edge.to_node
                visited = set()
                increment = self._calculate_increment(target_node, source_node, edge_values, spanning_tree, visited)
                if increment is None:  # Check if None
                    raise Exception("Unable to compute increment for chord edge")
                increment += edge_values[edge]
                increments[edge] = increment

        return increments
    
    def _dfs_extract_paths(self, current_node, modified_graph: List[Edge], 
                          back_edges: List[Edge], increments: Dict, current_increment: int, 
                          paths: Dict, current_path: List[Edge]) -> None:
        """DFS path extraction"""
        outgoing_edges = get_outgoing_edges(current_node, modified_graph)
        
        if len(outgoing_edges) == 0:
            paths[current_increment] = {
                'is_loop_path': False,
                'path': current_path
            }
        else:
            for edge in outgoing_edges:
                new_increment = current_increment + increments[edge]
                
                if edge.back_edge_entry_mapping is not None:
                    new_path = [edge]
                    self._dfs_extract_paths(
                        edge.to_node, modified_graph, back_edges, 
                        increments, increments[edge], paths, new_path
                    )
                elif edge.back_edge_exit_mapping is not None:
                    new_path = list(current_path)
                    new_path.append(edge.back_edge_exit_mapping)
                    paths[new_increment] = {
                        'is_loop_path': True,
                        'path': new_path
                    }
                else:
                    new_path = list(current_path)
                    new_path.append(edge)
                    self._dfs_extract_paths(
                        edge.to_node, modified_graph, back_edges, 
                        increments, new_increment, paths, new_path
                    )
    
    def run_epp_profiling(self, extended_contracts_graph_with_entry_exit) -> Tuple[List[Edge], List[Edge], Dict]:
        """
        Execute Efficient Path Profiling (EPP) algorithm
        
        Args:
            extended_contracts_graph_with_entry_exit: Extended contract graph with entry and exit nodes
            
        Returns:
            Tuple[List[Edge], List[Edge], Dict]: 
            - Back edges list
            - Modified graph (removed back edges and added loops)
            - Increments dictionary
        """
        graph = extended_contracts_graph_with_entry_exit.edges
        vertices = extended_contracts_graph_with_entry_exit.vertices

        self._verify_graph_structure(vertices, graph)

        entry = get_entry(vertices, graph)
        exit_node = get_exit(vertices, graph)

        # Find back edges
        color = defaultdict(int)
        back_edges = []
        self._dfs_find_back_edges(entry, graph, color, back_edges)

        # Build modified graph with back edges removed and loops added
        modified_graph = []
        for edge in graph:
            is_back_edge = edge in back_edges

            if is_back_edge:
                # Add entry to head edge
                entry_to_head = Edge(entry, edge.to_node)
                entry_to_head.back_edge_entry_mapping = edge
                modified_graph.append(entry_to_head)

                # Add tail to exit edge
                tail_to_exit = Edge(edge.from_node, exit_node)
                tail_to_exit.back_edge_exit_mapping = edge
                modified_graph.append(tail_to_exit)
            else:
                modified_graph.append(edge)

        # Topological sort
        sorted_nodes = self._topological_sort(vertices, modified_graph)

        # Generate edge values
        edge_values = self._generate_epp_edge_values(sorted_nodes, modified_graph)

        # Build complete graph
        complete_graph = list(modified_graph)
        exit_to_entry = Edge(exit_node, entry)
        edge_values[exit_to_entry] = 0
        complete_graph.append(exit_to_entry)

        # Generate maximum spanning tree
        spanning_tree = self._generate_maximum_spanning_tree(complete_graph)

        # Generate increments
        increments = self._generate_edge_increments(complete_graph, edge_values, spanning_tree)

        # Process entry and exit nodes
        roots_leaves_orphans = extended_contracts_graph_with_entry_exit.get_roots_leaves_orphans()
        special_vertices = roots_leaves_orphans['roots'].union(roots_leaves_orphans['orphans'])
        
        for vertex in special_vertices:
            for edge in graph:
                if edge.from_node is entry and edge.to_node is vertex and edge not in increments:
                    increments[edge] = 0

        special_vertices = roots_leaves_orphans['leaves'].union(roots_leaves_orphans['orphans'])
        for vertex in special_vertices:
            for edge in graph:
                if edge.from_node is vertex and edge.to_node is exit_node and edge not in increments:
                    increments[edge] = 0

        return back_edges, modified_graph, increments
    
    def run_pap_profiling(self, extended_contracts_graph_with_entry_exit) -> Tuple[List[Edge], Dict, Dict]:
        """
        Execute Path-aware Profiling (PAP) algorithm
        
        Args:
            extended_contracts_graph_with_entry_exit: Extended contract graph with entry and exit nodes
            
        Returns:
            Tuple[List[Edge], Dict, Dict]: 
            - Back edges list
            - s values dictionary (node in-degrees)
            - i values dictionary (edge indices)
            
        """
        graph = extended_contracts_graph_with_entry_exit.edges
        vertices = extended_contracts_graph_with_entry_exit.vertices

        self._verify_graph_structure(vertices, graph)
        
        entry = get_entry(vertices, graph)
        exit_node = get_exit(vertices, graph)

        # Find back edges
        color = defaultdict(int)
        back_edges = []
        self._dfs_find_back_edges(entry, graph, color, back_edges)

        if len(back_edges) == 0:
            raise Exception("No back edges found, no loop processing needed")

        # Count in-degrees and out-degrees
        vertices_with_in_out = []
        for vertex in vertices:
            node = Node(vertex)
            incoming_count = 0
            outgoing_count = 0
            incoming_edges = []
            outgoing_edges = []

            for edge in graph:
                if edge.to_node == vertex:
                    incoming_count += 1
                    incoming_edges.append(edge)
                if edge.from_node == vertex:
                    outgoing_count += 1
                    outgoing_edges.append(edge)
            
            node.node_in = incoming_count
            node.node_out = outgoing_count
            node.in_edges = incoming_edges
            node.out_edges = outgoing_edges
            vertices_with_in_out.append(node)

        # Generate PAP values
        s_values, i_values = self._generate_pap_edge_values(vertices_with_in_out)

        return back_edges, s_values, i_values
    
    def profile(self, extended_contracts_graph_with_entry_exit) -> Any:
        """
        Execute the corresponding path profiling algorithm based on configured type
        
        Returns:
            Different results based on algorithm type
        """
        if self.profiler_type == ProfilerType.EPP:
            return self.run_epp_profiling(extended_contracts_graph_with_entry_exit)
        elif self.profiler_type == ProfilerType.PAP:
            return self.run_pap_profiling(extended_contracts_graph_with_entry_exit)
        else:
            raise ValueError(f"Unsupported profiler type: {self.profiler_type}")

