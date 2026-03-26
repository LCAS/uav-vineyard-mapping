'''
Graph based approach to match a set of detected vineyard poles (represented by a graph) to a prior knowledge graph of expected pole locations. 
NetworkX optimal_edit_paths function finds the optimal sequence of edit operations (insertion, deletion, substitution) that transform one graph into another. 
This minimises the "edit distance" between the graphs, which is a measure of their dissimilarity.
The cost of substituting one node for another is based on the Euclidean distance between their positions.
Iteratively refines the matching by;
Finding the best node mapping using the current graph positions.
Estimating a transformation (rotation and translation) based on the matched node pairs.
Applying the transformation to the prior knowledge graph.
Repeating until convergence or a maximum number of iterations is reached.
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.patches import ConnectionPatch # for plotting matching result
from networkx.algorithms.similarity import graph_edit_distance
from typing import Dict, Tuple, Optional
from networkx.algorithms.similarity import optimize_graph_edit_distance, optimal_edit_paths
from pprint import pprint
import geojson
import math

import spatial_interpolation_stright_poles_only

# Prior knowledge of vineyard pole locations
# Create the prior knowledge graph

# Riseholeme
# num_rows = 10
# num_cols = 4 # number of poles per row
# row_spacing = 2.75 # meters between rows
# col_spacing = 5.65 # pole spacing meters along the row

# Riseholeme
# num_rows = 4 # number of poles per row
# num_cols = 10 # number of rows
# row_spacing = 5.65 # pole spacing meters along the row
# col_spacing = 2.75 # meters between rows
# num_poles_per_col = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4] # Riseholme

# JoJos first 10 rows from the west end
# row_spacing = 5.00 # pole spacing meters along the row
# col_spacing = 3.00 # meters between rows
# num_poles_per_col = [5, 7, 8, 9, 10, 11, 13, 14, 15, 16]

# Distance threshold for geojson grpah edge distances
# distance_threshold_max = 6.5 # m
# distance_threshold_min = 5.0 # m

# Riseholme
geojson_file = '../../data/clustered_poles_edited.geojson'
geojson_file_prior_knowledge = '../../data/rtk_gps_poles.geojson'

# JoJo
# geojson_file = '../../data/jojo_row_posts_10_rows.geojson'

node_ins_cost_value = 0.3
node_del_cost_value = 0.3

max_iterations_value = 10
convergence_threshold_value = 0.0001



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

def create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing):
    """
    Creates a graph where each column can have a different number of poles (nodes).

    Args:
        num_poles_per_col (list or int): A list specifying the number of poles in each column,
                                         or a single integer for a uniform number of poles per column.
        row_spacing (float): Spacing between rows (y-axis).
        col_spacing (float): Spacing between columns (x-axis).

    Returns:
        G (networkx.Graph): A graph with nodes placed in a layout with varying poles per column.
    """
    G = nx.Graph()

    # Ensure num_poles_per_col is a list
    if isinstance(num_poles_per_col, int):
        num_poles_per_col = [num_poles_per_col] * int(col_spacing / row_spacing)

    for col, num_rows in enumerate(num_poles_per_col):
        for row in range(num_rows):
            x = col * col_spacing
            y = row * row_spacing
            name = f"Node_{row}_{col}"  # Assign a name based on the row and column
            node = (x, y)
            G.add_node(node, pos=(x, y), name=name)

            # Connect nodes within the same column
            if row > 0:  # Connect to the node above (same column)
                G.add_edge(((col) * col_spacing, (row - 1) * row_spacing), node)

            # # Connect nodes within the same row (if a node exists to the left in this row)
            # if col > 0 and row < num_poles_per_col[col - 1]:  # Ensure row exists in the previous column
            #     G.add_edge(((col - 1) * col_spacing, row * row_spacing), node)

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
    
    # First graph - a simple connected structure (prior knowledge)
    # G1 = create_prior_knowledge_graph(num_rows, num_cols, row_spacing, col_spacing) # in long, lat coordinates
    # G1 = create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing) # in long, lat coordinates with variable poles per row
    G1 = create_detection_graph_cartesian(geojson_file_prior_knowledge) # in x, y coordinates

    positions1 = nx.get_node_attributes(G1, 'pos')

    # Second graph - similar but with small differences (generated from perception)
    # G2 = create_detection_graph(geojson_file) # in long, lat coordinates
    G2 = create_detection_graph_cartesian(geojson_file) # in x, y coordinates

    positions2 = nx.get_node_attributes(G2, 'pos')

    return G1, G2, positions1, positions2

def visualize_graphs(file_name, G1: nx.Graph, G2: nx.Graph, pos1: Dict, pos2: Dict) -> None:
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

    plt.savefig(file_name)
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
    # print(f"Node mappings: {node_mapping}")
    matched_nodes = []  # For visualizing matched nodes
    for n1, n2 in node_mapping:
        if n1 is None:
            # print(f"Node {n2} in G2 was inserted into G1")
            pos2 = G2.nodes[n2].get('pos', (0, 0))
            new_graph.add_node(n2, pos=pos2)
        elif n2 is None:
            # print(f"Node {n1} in G1 was deleted")
            continue
        else:
            pos1 = G1.nodes[n1].get('pos', (0, 0))
            pos2 = G2.nodes[n2].get('pos', (0, 0))
            dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
            # print(f"Node {n1} in G1 mapped to {n2} in G2 (distance: {dist:.3f})")
            matched_nodes.append((pos1, pos2))  # Store both positions (from G1 and G2)
            new_graph.add_node(pos1, pos=pos1) # Add matched node to the new graph

    # Visualize the graphs
    fig, ax = plt.subplots(figsize=(12, 8))

    # Combine positions into a single plot
    pos1 = nx.get_node_attributes(new_graph, 'pos')
    pos2 = nx.get_node_attributes(G2, 'pos')

    shift_x = 25  # Shift value for x-coordinate
    shift_y = 2  # Shift value for y-coordinate

    # Shift both x and y coordinates
    pos2_shifted = {k: (v[0] + shift_x, v[1] + shift_y) for k, v in pos2.items()}

    # Define node color map based on matching for G1
    node_colors_G1 = []
    for node in new_graph.nodes:
        if any(node == pos[0] or node == pos[1] for pos in matched_nodes):  # If node is matched
            node_colors_G1.append('cyan')  # Color for matched nodes in G1
            # print(f"Node {node} is matched in G1")
        else:
            node_colors_G1.append('blue')  # Color for unmatched nodes in G1

    # Define node color map for G2
    node_colors_G2 = []
    for node in G2.nodes:
        # Check if the node's position in G2 is part of the matched_nodes
        if any(node == pos[1] for pos in matched_nodes):  # If node is matched in G2
            node_colors_G2.append('gray')  # Color for matched nodes in G2
            # print(f"Node {node} is matched in G2")
        else:
            node_colors_G2.append('green')  # Color for unmatched nodes in G2

    # Draw new_graph
    nx.draw(
        new_graph, pos1, ax=ax, with_labels=False, node_color=node_colors_G1, node_size=500, label="Graph G1", font_size=4
    )

    # Draw G2 (shifted to avoid overlap)
    nx.draw(
        G2, pos2_shifted, ax=ax, with_labels=False, node_color=node_colors_G2, node_size=200, label="Graph G2", font_size=4
    )

    # Draw red lines between matched nodes from G1 and G2
    for pos1, pos2 in matched_nodes:
        # Shift the pos2 node positions as done earlier
        shifted_pos2 = (pos2[0] + shift_x, pos2[1] + shift_y)
        ax.plot([pos1[0], shifted_pos2[0]], [pos1[1], shifted_pos2[1]], color='red', linestyle='--', linewidth=1)

    # Define a red dashed line for the legend
    red_line = mlines.Line2D([], [], color='red', linestyle='--', linewidth=1, label='Matched Nodes')
    blue_circle = mlines.Line2D([], [], color='blue', linestyle='None', marker='o', markersize=8, label='New Graph (Inserted Nodes)')
    green_circle = mlines.Line2D([], [], color='green', linestyle='None', marker='o', markersize=8, label='Prior Knowledge Graph')
    cyan_circle = mlines.Line2D([], [], color='cyan', linestyle='None', marker='o', markersize=8, label='Matched Nodes in New Graph')
    grey_circle = mlines.Line2D([], [], color='gray', linestyle='None', marker='o', markersize=8, label='Matched Nodes in Prior Knowledge Graph')

    # Add the custom legend to the plot
    ax.legend(handles=[red_line, blue_circle, green_circle, cyan_circle, grey_circle], loc="upper left", bbox_to_anchor=(1, 1))

    # Adjust the layout to prevent clipping
    plt.tight_layout()

    # Title and display
    ax.set_title("Graph Matching Visualization")
    plt.show()

    plt.savefig('../../images/graph_matching/gmatch4py/3_gmatch4py_graph_matching.png')
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
            node_ins_cost=lambda x: node_ins_cost_value,
            node_del_cost=lambda x: node_del_cost_value,
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
        
import networkx as nx
import numpy as np
from networkx.algorithms.similarity import optimize_graph_edit_distance
from typing import Dict, Tuple, List, Optional
from scipy.spatial import transform
from math import atan2

def get_node_positions(G: nx.Graph) -> np.ndarray:
    """Extract node positions as numpy array"""
    positions = nx.get_node_attributes(G, 'pos')
    return np.array([pos for pos in positions.values()])

def estimate_transformation(source_points: np.ndarray, target_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate rotation matrix and translation vector between two point sets
    Returns: (rotation_matrix, translation_vector)
    """
    # Center the points
    source_center = np.mean(source_points, axis=0)
    target_center = np.mean(target_points, axis=0)

    # Center both point sets
    source_centered = source_points - source_center
    target_centered = target_points - target_center

    # Calculate rotation using SVD
    H = source_centered.T @ target_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure we have a rotation matrix (det=1)
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T

    # Calculate translation
    t = target_center - R @ source_center

    return R, t

def apply_transformation(G: nx.Graph, R: np.ndarray, t: np.ndarray) -> nx.Graph:
    """Apply rotation and translation to graph nodes"""
    G_transformed = G.copy()
    pos = nx.get_node_attributes(G, 'pos')

    for node in G.nodes():
        point = np.array(pos[node])
        transformed_point = R @ point + t
        G_transformed.nodes[node]['pos'] = tuple(transformed_point)

    return G_transformed

def node_subst_cost_weighted(node1_data: Dict, node2_data: Dict, position_weight: float = 1.0) -> float:
    """
    Calculate node substitution cost with adjustable position weight
    """
    pos1 = np.array(node1_data.get('pos', (0, 0)))
    pos2 = np.array(node2_data.get('pos', (0, 0)))
    distance = np.linalg.norm(pos1 - pos2)
    return position_weight * distance

def iterative_graph_matching(G1: nx.Graph, G2: nx.Graph,
                           max_iterations: int = 10,
                           convergence_threshold: float = 0.0001) -> Tuple[nx.Graph, List, float]:
    """
    Iteratively match graphs and estimate transformation
    Returns: (transformed_G1, best_node_mapping, final_distance)
    """
    current_G1 = G1.copy()
    best_distance = float('inf')
    best_mapping = None
    best_transformed_G1 = None

    global_R=np.eye(2)
    global_t=np.zeros(2)

    for iteration in range(max_iterations):
        # 1. Find best node mapping with current positions
        try:
            associations = optimal_edit_paths (
                current_G1, G2,
                node_subst_cost=lambda x,y: node_subst_cost(x,y)*1,
                edge_subst_cost=lambda x,y: edge_subst_cost(x,y)*1,
                # node_ins_cost=lambda x: 1.0 + 0.5 * x.get('weight', 0),
                # node_del_cost=lambda x: 1.0 + 0.5 * x.get('weight', 0),
                node_ins_cost=lambda x: node_ins_cost_value,
                node_del_cost=lambda x: node_del_cost_value,
                edge_ins_cost=lambda x: 0, # as we dind't have edges in the original graph, we allow additions for free
                edge_del_cost=lambda x: 1,
            )

            nodes_pairs = associations[0][0][0]
            current_distance = associations[-1]

            # Extract matched points
            matched_pairs = []
            for op in nodes_pairs:
                if op[0] and op[1]:
                    # match
                    source_pos = np.array(current_G1.nodes[op[0]]['pos'])
                    target_pos = np.array(G2.nodes[op[1]]['pos'])
                    matched_pairs.append((source_pos, target_pos))
            #print(f'matched {len(matched_pairs)} pairs')
            if not matched_pairs:
                print("No valid matches found")
                break

            # 2. Estimate transformation
            source_points = np.array([p[0] for p in matched_pairs])
            target_points = np.array([p[1] for p in matched_pairs])

            R, t = estimate_transformation(source_points, target_points)

            #print(R, t)

            # 3. Apply transformation
            current_G1 = apply_transformation(current_G1, R, t)

            # Update best result if improved
            if current_distance < best_distance:
                best_distance = current_distance
                best_mapping = nodes_pairs
                best_transformed_G1 = current_G1.copy()
            else:
                # If no improvement, consider convergence
                improvement = (best_distance - current_distance) / best_distance
                if abs(improvement) < convergence_threshold:
                    print(f"Converged after {iteration + 1} iterations")
                    break

            print(f"Iteration {iteration + 1}: Distance = {current_distance:.3f}")

        except (nx.NetworkXError, StopIteration) as e:
            print(f"Error in iteration {iteration + 1}: {e}")
            break


    # calculate the final transformation of all nodes from G1
    matched_pairs = []
    for n in G1.nodes:
            source_pos = np.array(G1.nodes[n]['pos'])
            target_pos = np.array(current_G1.nodes[n]['pos'])

            matched_pairs.append((source_pos, target_pos))

    if not matched_pairs:
        print("No valid matches found")

    # 2. Estimate transformation
    source_points = np.array([p[0] for p in matched_pairs])
    target_points = np.array([p[1] for p in matched_pairs])

    global_R, global_t = estimate_transformation(source_points, target_points)



    return best_transformed_G1, best_mapping, best_distance, global_R, global_t

def main():
    # Create sample graphs
    G1, G2, pos1, pos2 = create_sample_topological_graphs()

    # Visualize the graphs
    visualize_graphs('../../images/graph_matching/gmatch4py/1_gmatch4py_graphs.png', G2, G1, pos2, pos1)

    # Compare the graphs
    node_mapping = compare_topological_graphs(G2, G1)

    # print(f"Mapping: {node_mapping}")

    analyze_differences(G2, G1, node_mapping[0])

    transformed_G2, best_mapping, final_distance, R, t = iterative_graph_matching(G2, G1, max_iterations_value, convergence_threshold_value)

    print(f"final best mapping: {best_mapping} with a distance of {final_distance}")
    print(f"the final roation was {atan2(R[0,1],R[0,0]) * 180.0 / math.pi} degrees, t was {t}")

    analyze_differences(transformed_G2, G1, best_mapping)
    new_poses={}
    for n in transformed_G2.nodes:
        pos = transformed_G2.nodes[n].get('pos')
        new_poses[n]=pos
    #print(transformed_G2.nodes[])

    # print('Original problem:')
    # visualize_graphs('../../images/graph_matching/gmatch4py/1_gmatch4py_graphs.png', G2, G1, pos2, pos1)
    # print('After optimisation:')
    visualize_graphs('../../images/graph_matching/gmatch4py/2_gmatch4py_graphs_optimised.png', transformed_G2, G1, new_poses, pos1)

if __name__ == "__main__":
    main()