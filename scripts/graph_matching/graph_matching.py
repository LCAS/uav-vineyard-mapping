import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from networkx.algorithms.similarity import graph_edit_distance
from typing import Dict, Tuple, Optional
from networkx.algorithms.similarity import optimize_graph_edit_distance, optimal_edit_paths
from pprint import pprint
import geojson
import math

# Prior knowledge of vineyard pole locations
# Create the prior knowledge graph
# num_rows = 10
# num_cols = 4 # number of poles per row
# row_spacing = 2.75 # meters between rows
# col_spacing = 5.65 # pole spacing meters along the row

num_rows = 4 # number of poles per row
num_cols = 10 # number of rows
row_spacing = 5.65 # pole spacing meters along the row
col_spacing = 2.75 # meters between rows

# Distance threshold
distance_threshold_max = 6.5 # m
distance_threshold_min = 5.0 # m

def create_prior_knowledge_graph(num_rows, num_cols, row_spacing, col_spacing):
    """
    Creates a grid graph where nodes are placed in a regular grid layout
    with specified spacing between rows and columns, and only connects nodes 
    that are in the same row or column.

    Args:
        num_rows (int): Number of rows in the grid.
        num_cols (int): Number of columns in the grid.
        row_spacing (float): Spacing between rows (y-axis).
        col_spacing (float): Spacing between columns (x-axis).

    Returns:
        G (networkx.Graph): A grid graph.
    """
    G = nx.Graph()

    for row in range(num_rows):
        for col in range(num_cols):
            x = col * col_spacing
            y = row * row_spacing
            name = f"Node_{row}_{col}"  # Assign a name based on the row and column
            node = (x, y)
            G.add_node(node, pos=(x, y), name=name)

            # # Connect nodes only within the same row
            # if col > 0:  # Connect to the node to the left (same row)
            #     G.add_edge(((col - 1) * col_spacing, row * row_spacing), node)

            # Connect nodes only within the same column
            if row > 0:  # Connect to the node above (same column)
                G.add_edge(((col) * col_spacing, (row - 1) * row_spacing), node)

    return G

def create_detection_graph_cartesian(geojson_file):
    """
    Converts GeoJSON coordinates (longitude, latitude) in a graph to Cartesian coordinates (x, y),
    where the southwest-most node becomes the origin (0, 0).

    Args:
        geojson_file (str): Path to the GeoJSON file with the pole locations.

    Returns:
        G (networkx.Graph): The resulting graph with Cartesian coordinates.
    """
    with open(geojson_file) as f:
        data = geojson.load(f)

    G = nx.Graph()

    # Add nodes and find the southwest-most coordinate
    coords_list = []
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        coords_list.append(coords)
        G.add_node(coords, pos=coords)  # Assign original coordinates as a node attribute

    # Southwest-most node as the origin
    min_lon, min_lat = min(coords_list, key=lambda c: (c[1], c[0]))

    # Conversion to Cartesian coordinates
    def latlon_to_xy(lon, lat, origin_lon, origin_lat):
        """Converts (lon, lat) to (x, y) relative to an origin point."""
        from pyproj import Proj, transform

        # Define projection: WGS84 to a local projected system
        proj_wgs84 = Proj(proj='latlong', datum='WGS84')
        proj_local = Proj(proj='aeqd', lat_0=origin_lat, lon_0=origin_lon)  # Azimuthal Equidistant

        x, y = transform(proj_wgs84, proj_local, lon, lat)
        return x, y

    # Replace coordinates in the graph
    for node in list(G.nodes()):
        lon, lat = node
        x, y = latlon_to_xy(lon, lat, min_lon, min_lat)
        nx.set_node_attributes(G, {node: (x, y)}, 'pos')
        G = nx.relabel_nodes(G, {node: (x, y)})

    # # Add edges based on proximity (adjust distance threshold as needed)
    # for u in G.nodes():
    #     for v in G.nodes():
    #         if u != v:
    #             # Calculate Euclidean distance between the two nodes
    #             distance = math.sqrt((u[0] - v[0])**2 + (u[1] - v[1])**2)
    #             if distance < distance_threshold_max and distance > distance_threshold_min:
    #                 G.add_edge(u, v)

    return G

def haversine_distance(coord1, coord2):
    """
    Calculates the Haversine distance between two geographic coordinates.

    Args:
        coord1: A tuple of (latitude1, longitude1) in decimal degrees.
        coord2: A tuple of (latitude2, longitude2) in decimal degrees.

    Returns:
        The distance between the two points in kilometers.
    """

    R = 6371  # Earth radius in kilometers

    lat1, lon1 = coord1
    lat2, lon2 = coord2

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    distance = R * c
    # print("Distance between", coord1, "and", coord2, "is", distance, "km")
    return distance

def create_detection_graph(geojson_file):
    """Creates a graph from detected pole locations in a GeoJSON file."""
    with open(geojson_file) as f:
        data = geojson.load(f)

    # # Create the new point feature for missing pole
    # new_feature = {
    #     "type": "Feature",
    #     "geometry": {
    #         "type": "Point",
    #         "coordinates": [-0.524587, 53.26805967]
    #     },
    #     "properties": {
    #         "type": "pole",
    #         "pole_id": 39  # Assign a new unique ID
    #     }
    # }

    # # Add the new feature to the GeoJSON data
    # data['features'].append(new_feature)

    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, pos=coords)  # Assign coordinates as node attribute

    # # Add edges based on proximity (adjust distance threshold as needed)
    # for u in G.nodes():
    #     for v in G.nodes():
    #         if u != v and haversine_distance(u, v) < distance_threshold_max / 1000 and haversine_distance(u, v) > distance_threshold_min / 1000:
    #             # print("Distance threshold is set to:", distance_threshold)
    #             # print("Adding edge between", u, "and", v)
    #             G.add_edge(u, v)

    return G

def create_sample_topological_graphs() -> Tuple[nx.Graph, nx.Graph, Dict, Dict]:
    """
    Creates two sample topological graphs with slightly different structures
    Returns: (graph1, graph2, pos1, pos2)
    """
    
    geojson_file = '../../data/clustered_poles.geojson'

    # First graph - a simple connected structure (prior knowledge)
    G1 = create_prior_knowledge_graph(num_rows, num_cols, row_spacing, col_spacing) # in long, lat coordinates

    positions1 = nx.get_node_attributes(G1, 'pos')

    # Second graph - similar but with small differences (generated from perception)
    # G2 = create_detection_graph(geojson_file) # in long, lat coordinates
    G2 = create_detection_graph_cartesian(geojson_file) # in x, y coordinates

    positions2 = nx.get_node_attributes(G2, 'pos')

    return G1, G2, positions1, positions2

def visualize_graphs(G1: nx.Graph, G2: nx.Graph, pos1: Dict, pos2: Dict) -> None:
    """
    Visualize both graphs side by side for comparison
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Draw first graph
    nx.draw(G1, pos1, ax=ax1, with_labels=False, node_color='blue', node_size=500, font_size=8)
    ax1.set_title("Graph 1 (perceived)")

    # Draw second graph
    nx.draw(G2, pos2, ax=ax2, with_labels=False, node_color='green', node_size=500, font_size=8)
    ax2.set_title("Graph 2 (prior to match to)")

    plt.tight_layout()
    plt.show()

    plt.savefig('../../images/graph_matching/gmatch4py/1_gmatch4py_graphs.png')
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
    
    new_graph = nx.Graph()

    # Analyze node differences
    print(f"Node mappings: {node_mapping}")
    matched_edges = [] # For visualizing matched nodes
    matched_nodes = []  # For visualizing matched nodes
    for n1, n2 in node_mapping:
        if n1 is None:
            print(f"Node {n2} in G2 was inserted into G1")
            pos2 = G2.nodes[n2].get('pos', (0, 0))
            new_graph.add_node(n2, pos=pos2)
        elif n2 is None:
            print(f"Node {n1} in G1 was deleted")
        else:
            pos1 = G1.nodes[n1].get('pos', (0, 0))
            pos2 = G2.nodes[n2].get('pos', (0, 0))
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            print(f"Node {n1} in G1 mapped to {n2} in G2 (distance: {dist:.3f})")
            matched_edges.append((n1, n2))  # Add matching pair
            matched_nodes.append((pos1, pos2))  # Store both positions (from G1 and G2)
            new_graph.add_node(pos1, pos=pos1) # Add matched node to the new graph

    # Visualize the graphs
    fig, ax = plt.subplots(figsize=(12, 8))

    # Combine positions into a single plot
    pos1 = nx.get_node_attributes(new_graph, 'pos')
    pos2 = nx.get_node_attributes(G2, 'pos')
    pos2_shifted = {k: (v[0] + 5, v[1]) for k, v in pos2.items()}  # Shift G2 positions

    # Define node color map based on matching for G1
    node_colors_G1 = []
    for node in new_graph.nodes:
        if any(node == pos[0] or node == pos[1] for pos in matched_nodes):  # If node is matched
            node_colors_G1.append('red')  # Color for matched nodes in G1
            print(f"Node {node} is matched in G1")
        else:
            node_colors_G1.append('blue')  # Color for unmatched nodes in G1

    # Define node color map for G2
    node_colors_G2 = []
    for node in G2.nodes:
        # Check if the node's position in G2 is part of the matched_nodes
        if any(node == pos[1] for pos in matched_nodes):  # If node is matched in G2
            node_colors_G2.append('gray')  # Color for matched nodes in G2
            print(f"Node {node} is matched in G2")
        else:
            node_colors_G2.append('green')  # Color for unmatched nodes in G2

    # Draw new_graph
    nx.draw(
        new_graph, pos1, ax=ax, with_labels=False, node_color=node_colors_G1, node_size=500, label="Graph G1", font_size=4
    )

    # Draw G2 (shifted to avoid overlap)
    nx.draw(
        G2, pos2, ax=ax, with_labels=False, node_color=node_colors_G2, node_size=200, label="Graph G2", font_size=4
    )

    # Draw matched edges
    for n1, n2 in matched_edges:
        if n1 in pos1 and n2 in pos2:
            x1, y1 = pos1[n1]
            x2, y2 = pos2_shifted[n2]
            ax.plot([x1, x2 + 5], [y1, y2], color="red", linestyle="--", linewidth=0.5)
            print(f"Matched edge: {n1} -> {n2}")

    # Define a red dashed line for the legend
    red_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=2, label='Matched Nodes')
    blue_circle = mlines.Line2D([], [], color='blue', linestyle='None', marker='o', markersize=8, label='New Graph')
    green_circle = mlines.Line2D([], [], color='green', linestyle='None', marker='o', markersize=8, label='Prior Knowledge Graph')
    red_circle = mlines.Line2D([], [], color='red', linestyle='None', marker='o', markersize=8, label='Matched Nodes in New Graph')
    grey_circle = mlines.Line2D([], [], color='gray', linestyle='None', marker='o', markersize=8, label='Matched Nodes in Prior Knowledge Graph')

    # Add the custom legend to the plot
    ax.legend(handles=[red_line, blue_circle, green_circle, red_circle, grey_circle], loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust the layout to prevent clipping
    plt.tight_layout()

    # Title and display
    ax.set_title("Graph Matching Visualization")
    plt.show()

    plt.savefig('../../images/graph_matching/gmatch4py/2_gmatch4py_graph_matching.png')
    plt.close()

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