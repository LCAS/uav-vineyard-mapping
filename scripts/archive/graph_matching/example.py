import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import geojson

def create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing, rotation_angle=0):
    """
    Creates a graph where each column can have a different number of poles (nodes) and applies a rotation.

    Args:
        num_poles_per_col (list or int): A list specifying the number of poles in each column,
                                         or a single integer for a uniform number of poles per column.
        row_spacing (float): Spacing between rows (y-axis).
        col_spacing (float): Spacing between columns (x-axis).
        rotation_angle (float): Rotation angle in degrees.

    Returns:
        G (networkx.Graph): A graph with nodes placed in a layout with varying poles per column.
    """
    G = nx.Graph()

    # Ensure num_poles_per_col is a list
    if isinstance(num_poles_per_col, int):
        num_poles_per_col = [num_poles_per_col] * int(col_spacing / row_spacing)

    # Convert rotation angle to radians
    theta = np.radians(rotation_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    for col, num_rows in enumerate(num_poles_per_col):
        for row in range(num_rows):
            x = col * col_spacing
            y = row * row_spacing

            # Apply rotation
            x_rot = x * cos_theta - y * sin_theta
            y_rot = x * sin_theta + y * cos_theta

            name = f"Node_{row}_{col}"  # Assign a name based on the row and column
            node = (x_rot, y_rot)
            G.add_node(node, pos=(x_rot, y_rot), name=name)

            # Connect nodes within the same column
            if row > 0:  # Connect to the node above (same column)
                prev_node = (col * col_spacing * cos_theta - (row - 1) * row_spacing * sin_theta,
                             col * col_spacing * sin_theta + (row - 1) * row_spacing * cos_theta)
                G.add_edge(prev_node, node)

            # # Connect nodes within the same row (if a node exists to the left in this row)
            # if col > 0 and row < num_poles_per_col[col - 1]:  # Ensure row exists in the previous column
            #     prev_col_node = ((col - 1) * col_spacing * cos_theta - row * row_spacing * sin_theta,
            #                      (col - 1) * col_spacing * sin_theta + row * row_spacing * cos_theta)
            #     G.add_edge(prev_col_node, node)

    return G

def create_detection_graph_cartesian(geojson_file, angle_range):
    """
    Converts GeoJSON coordinates (longitude, latitude) in a graph to Cartesian coordinates (x, y),
    where the southwest-most node becomes the origin (0, 0). Adds edges based on angle criteria.

    Args:
        geojson_file (str): Path to the GeoJSON file with the pole locations.
        angle_range (tuple): Tuple of (min_angle, max_angle) in degrees to consider for adding an edge.

    Returns:
        G (networkx.Graph): The resulting graph with Cartesian coordinates and edges.
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

    # Add edges based on angle criteria
    nodes = list(G.nodes())
    min_angle, max_angle = angle_range
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes):
            if i != j:
                pos1 = np.array(G.nodes[node1]['pos'])
                pos2 = np.array(G.nodes[node2]['pos'])
                angle = np.degrees(np.arctan2(pos2[1] - pos1[1], pos2[0] - pos1[0]))
                if min_angle <= angle <= max_angle:
                    G.add_edge(node1, node2)

    return G

# Riseholme
# row_spacing = 5.65  # pole spacing meters along the row
# col_spacing = 2.75  # meters between rows
# num_poles_per_col = [4, 4, 4, 4, 4, 4, 4, 4, 4, 4]
# geojson_file = '../../data/clustered_poles.geojson'

# JoJos first 10 rows from the west end
row_spacing = 5.00 # pole spacing meters along the row
col_spacing = 3.00 # meters between rows
num_poles_per_col = [5, 7, 8, 9, 10, 11, 13, 14, 15, 16]
geojson_file = '../../data/jojo_row_posts_10_rows.geojson'

G1 = create_prior_knowledge_graph_with_variable_poles(num_poles_per_col, row_spacing, col_spacing, 295)  # in long, lat coordinates with variable poles per row
pos1 = nx.get_node_attributes(G1, 'pos')

plt.figure(figsize=(10, 8))
nx.draw(G1, pos=pos1, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=500)
plt.title("Prior Graph")
plt.savefig("1_prior_graph.png")

# Second graph - similar but with small differences (generated from perception)
G2 = create_detection_graph_cartesian(geojson_file, (35, 38))  # in x, y coordinates
pos = nx.get_node_attributes(G2, 'pos')

plt.figure(figsize=(10, 8))
nx.draw(G2, pos=pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=500)
plt.title("Detection Graph")
plt.savefig("2_detection_graph.png")

# Add dummy nodes to the smaller graph
num_dummy_nodes = abs(len(G1.nodes) - len(G2.nodes))
if len(G1.nodes) < len(G2.nodes):
    for i in range(num_dummy_nodes):
        dummy_node = f'dummy_{i}'
        G1.add_node(dummy_node)
        pos[dummy_node] = (0, 0)  # Assign a default position for dummy nodes
elif len(G2.nodes) < len(G1.nodes):
    for i in range(num_dummy_nodes):
        dummy_node = f'dummy_{i}'
        G2.add_node(dummy_node)
        pos[dummy_node] = (0, 0)  # Assign a default position for dummy nodes

# Ensure all nodes in G1 have positions
for node in G1.nodes:
    if node not in pos:
        pos[node] = node  # Use the node's coordinates as its position

# Create a cost matrix based on distances
cost_matrix = np.zeros((len(G1.nodes), len(G2.nodes)))
for i, node1 in enumerate(G1.nodes):
    for j, node2 in enumerate(G2.nodes):
        cost_matrix[i, j] = np.linalg.norm(np.array(pos[node2]) - np.array(pos[node1]))

print("Cost matrix:\n", cost_matrix)

# Solve the assignment problem
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Create a mapping from G1 to G2
mapping = {list(G1.nodes)[i]: list(G2.nodes)[j] for i, j in zip(row_ind, col_ind) if not isinstance(list(G1.nodes)[i], str) and not isinstance(list(G2.nodes)[j], str)}

print("Node mapping from G1 to G2:", mapping)

# Add edges from G1 to G2 based on the mapping, ensuring nodes have a maximum of two edges
for g1_node, g2_node in mapping.items():
    for neighbor in G1.neighbors(g1_node):
        if neighbor in mapping and G2.degree[g2_node] < 2 and G2.degree[mapping[neighbor]] < 2:
            G2.add_edge(g2_node, mapping[neighbor])

# Visualize the updated detection graph with edges from prior knowledge graph
plt.figure(figsize=(10, 8))
nx.draw(G2, pos=pos, with_labels=False, node_color='lightgreen', edge_color='gray', node_size=500)
plt.title("Updated Detection Graph with Edges from Prior Knowledge Graph")
plt.savefig("3_updated_detection_graph.png")

print("The updated detection graph with edges from the prior knowledge graph has been visualized and saved as 'updated_detection_graph.png'.")