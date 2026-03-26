import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from networkx.algorithms.similarity import graph_edit_distance
from typing import Dict, Tuple, Optional
from networkx.algorithms.similarity import optimize_graph_edit_distance, optimal_edit_paths
from pprint import pprint

def create_sample_topological_graphs() -> Tuple[nx.Graph, nx.Graph, Dict, Dict]:
    """
    Creates two sample topological graphs with slightly different structures
    Returns: (graph1, graph2, pos1, pos2)
    """
    # First graph - a simple connected structure (prior knowledge)
    G1 = nx.Graph()
    # Add nodes with positions
    positions1 = {
        'A': (0, -1),
        'B': (1, -1),
        'C': (2, -1),
        'D': (0, 0),
        'E': (1, 0),
        'F': (2, 0),
        'G': (0, 1),
        'H': (1, 1),
        'I': (2, 1)
    }
    for node, pos in positions1.items():
        G1.add_node(node, pos=pos)

    # Add edges to create a connected structure
    G1.add_edges_from([('A', 'B'), ('B', 'C'), ('D', 'E'), ('E', 'F'), ('G', 'H'), ('H', 'I')])

    # Second graph - similar but with small differences (generated from perception)
    G2 = nx.Graph()
    # Slightly shifted positions
    positions2 = {
        'A': (0, -1.1),
        #'B': (0.8, -1.2),       # B was missed
        'C': (2.1, -1.1),
        'D': (0.2, 0.3),
        'D1': (0.6, 0.33),      # some more noisy additional detections
        'D2': (0.65, 0.23),
        'D3': (0.55, 0.23),
        'E': (1.5, 0.5),
        #'F': (2.2, 0.4),          # F was missed
        'G': (0, 0.9),
        'G1': (0.2, 0.8),
        'G2': (0.6, 0.7),
        'H': (1, 1.1),
        'I': (2.1, 1.2)
    }
    for node, pos in positions2.items():
        G2.add_node(node, pos=pos)

    # Add edges with one different connection
    # We don't impose any structure in the pole positions, hence NO edges (we can add some where we are certain)
    #G2.add_edges_from([('A', 'B'), ('B', 'C'), ('Dfake', 'H'), ('D', 'Dfake'), ('E', 'F'), ('G', 'H'), ('E', 'I')])


    return G1, G2, positions1, positions2

def visualize_graphs(G1: nx.Graph, G2: nx.Graph, pos1: Dict, pos2: Dict) -> None:
    """
    Visualize both graphs side by side for comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Draw first graph
    nx.draw(G1, pos1, ax=ax1, with_labels=True, node_color='lightblue',
           node_size=500, font_size=16, font_weight='bold')
    ax1.set_title("Graph 1 (perceived)")

    # Draw second graph
    nx.draw(G2, pos2, ax=ax2, with_labels=True, node_color='lightgreen',
           node_size=500, font_size=16, font_weight='bold')
    ax2.set_title("Graph 2 (prior to match to)")

    plt.tight_layout()
    plt.show()

    plt.savefig('../../images/graph_matching/gmatch4py/gmatch4py_graph_matching_example.png')
    plt.close()

def node_subst_cost(node1_data: Dict, node2_data: Dict) -> float:
    """
    Calculate the cost of substituting one node for another based on their positions
    """
    pos1 = node1_data.get('pos', (0, 0))
    pos2 = node2_data.get('pos', (0, 0))
    # Euclidean distance between positions
    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    # Normalize the distance to give more weight to small positional differences
    return np.tanh(distance)

def edge_subst_cost(edge1_data: Dict, edge2_data: Dict) -> float:
    """
    Calculate the cost of substituting one edge for another
    In this case, we're using a simple constant cost
    """
    return 0.5

def analyze_differences(G1: nx.Graph, G2: nx.Graph, node_mapping: list) -> None:
    """
    Analyze and print the differences between the two graphs based on the GED mapping
    """
    if node_mapping is None:
        print("No valid mapping found")
        return

    # Analyze node differences
    print(f"Node mappings: {node_mapping}")
    for n1, n2 in node_mapping:
        if n1 is None:
            print(f"Node {n2} in G2 was inserted")
        elif n2 is None:
            print(f"Node {n1} in G1 was deleted")
        else:
            pos1 = G1.nodes[n1].get('pos', (0, 0))
            pos2 = G2.nodes[n2].get('pos', (0, 0))
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            print(f"Node {n1} mapped to {n2} (distance: {dist:.3f})")



def compare_topological_graphs(G1: nx.Graph, G2: nx.Graph) -> Tuple[float, list]:
    """
    Compare two topological graphs using graph edit distance
    Returns: (distance, best_node_mapping)
    """
    try:
        # Get all possible edit paths
        #edit_paths = optimize_graph_edit_distance(
        #    G1, G2,
        #    node_subst_cost=node_subst_cost,
        #    edge_subst_cost=edge_subst_cost
        #)
        associations = optimal_edit_paths (
            G1, G2,
            node_subst_cost=lambda x,y: node_subst_cost(x,y)*1,
            edge_subst_cost=lambda x,y: edge_subst_cost(x,y)*1,
            node_ins_cost=lambda x: 0.3,
            node_del_cost=lambda x: 0.3,
            edge_ins_cost=lambda x: 0, # as we dind't have edges in the original graph, we allow additions for free
            edge_del_cost=lambda x: 1,
        )
        pprint(list(associations))
        return associations[0][0]
        # Get the first (best) edit path
        best_edit_path = next(edit_paths)

        # Calculate the distance from the edit path
        distance = sum(1 for op in best_edit_path if op[0] != 'match')

        # Extract node mapping from the edit path
        node_mapping = []
        for operation in best_edit_path:
            op_type = operation[0]
            if op_type == 'substitute' or op_type == 'match':
                node_mapping.append((operation[1], operation[2]))
            elif op_type == 'delete':
                node_mapping.append((operation[1], None))
            elif op_type == 'insert':
                node_mapping.append((None, operation[1]))

        return distance, node_mapping

    except (nx.NetworkXError, StopIteration) as e:
        print(f"Error in comparison: {e}")
        return float('inf'), []


def main():
    # Create sample graphs
    G1, G2, pos1, pos2 = create_sample_topological_graphs()

    # Visualize the graphs
    visualize_graphs(G2, G1, pos2, pos1)

    # Compare the graphs
    node_mapping = compare_topological_graphs(G2, G1)

    print(f"Mapping: {node_mapping}")

    # Analyze and print differences
    analyze_differences(G2, G1, node_mapping[0])

if __name__ == "__main__":
    main()