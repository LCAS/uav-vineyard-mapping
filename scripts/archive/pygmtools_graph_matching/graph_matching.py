import geojson
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch  # for plotting matching result

import math

import numpy as np  # numpy backend
import pygmtools as pygm

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

# Distance threshold
distance_threshold_max = 6.5 /1000 # km
distance_threshold_min = 5.0 /1000 # km

def create_detection_graph(geojson_file):
    """Creates a graph from detected pole locations in a GeoJSON file."""
    with open(geojson_file) as f:
        data = geojson.load(f)

    G = nx.Graph()
    for feature in data['features']:
        coords = tuple(feature['geometry']['coordinates'])
        G.add_node(coords, pos=coords)  # Assign coordinates as node attribute

    # Add edges based on proximity (adjust distance threshold as needed)
    for u in G.nodes():
        for v in G.nodes():
            if u != v and haversine_distance(u, v) < distance_threshold_max and haversine_distance(u, v) > distance_threshold_min:
                # print("Distance threshold is set to:", distance_threshold)
                # print("Adding edge between", u, "and", v)
                G.add_edge(u, v)

    return G

def create_grid_graph(num_rows, num_cols, row_spacing, col_spacing):
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

def visualize_graph(G):
    """Visualizes the graph, preserving spatial relationships."""
    pos = nx.spring_layout(G, pos=nx.get_node_attributes(G, 'pos'))
    nx.draw_networkx(G, pos=pos, node_color='blue', node_size=100, with_labels=False)
    plt.axis('off')
    plt.savefig('../../images/pole_graph.png')

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)
    
geojson_file = '../../data/clustered_poles.geojson'

# Create the graph
G = create_detection_graph(geojson_file)

# Visualize and save the graph
visualize_graph(G)

G_prior = create_grid_graph(num_rows, num_cols, row_spacing, col_spacing)

for node, data in G_prior.nodes(data=True):
    print(f"Node: {node}, Position: {data['pos']}, Name: {data['name']}")

K = pygm.utils.build_aff_mat_from_networkx(G, G_prior)

X = pygm.rrwm(K, G.number_of_nodes(), G_prior.number_of_nodes())

X = pygm.hungarian(X)

# Visualize both graphs
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
pos1 = nx.get_node_attributes(G, 'pos')
nx.draw_networkx(G, pos=pos1, node_color='blue', node_size=100, with_labels=False)
plt.title("Detected Pole Graph")

plt.subplot(1, 2, 2)
pos2 = nx.get_node_attributes(G_prior, 'pos')
nx.draw_networkx(G_prior, pos=pos2, node_color='green', node_size=100, with_labels=False)
plt.title("Prior Knowledge Graph")
plt.savefig('../../images/pole_graph.png')

# print("pos1 keys:", pos1.keys())
# print("pos2 keys:", pos2.keys())

# Graph matching visualization
plt.figure(figsize=(12, 6))

# Create the subplots first
ax1 = plt.subplot(1, 2, 1)
plt.title('Detected Pole Graph')

# Create the second subplot
ax2 = plt.subplot(1, 2, 2)
plt.title('Prior Knowledge Graph')

number_of_nodes = 0

# Now add the connection lines (they will be drawn on top)
for i, node1 in enumerate(pos1.keys()):
    j = np.argmax(X[i]).item()  # Find the best match in `X`
    node2 = list(pos2.keys())[j]  # Get the corresponding key in `pos2`

    print(f"Matched node {node1} in G to node {node2} in G_prior")

    # Add a connection line between the two graphs
    con = ConnectionPatch(
        xyA=node1, 
        xyB=node2, 
        coordsA="data", 
        coordsB="data",
        axesA=ax1, 
        axesB=ax2, 
        color="green" if G_prior.has_edge(node1, node2) else "red",
        alpha=0.7
    )
    plt.gca().add_artist(con)
    
    number_of_nodes += 1

print("Number of nodes matched:", number_of_nodes)

# Draw the graph nodes first
nx.draw_networkx_nodes(G, pos=pos1, ax=ax1, node_size=50)
nx.draw_networkx_nodes(G_prior, pos=pos2, ax=ax2, node_size=50)

# Draw the graph edges last
nx.draw_networkx_edges(G, pos=pos1, ax=ax1)
nx.draw_networkx_edges(G_prior, pos=pos2, ax=ax2)

plt.suptitle("Graph Matching")
plt.savefig('../../images/pole_graph_matching.png')





# # Get the actual node orderings from both graphs
# nodes_G = list(G.nodes)  # List of nodes in G (detected pole graph)
# nodes_G_prior = list(G_prior.nodes)  # List of nodes in G_prior (prior knowledge graph)

# # Apply the graph matching result to align the second graph to the first one
# align_pos2 = {}
# for i in range(len(nodes_G)):  # Iterate over the nodes in G
#     j = np.argmax(X[i]).item()  # Find the best match in X
#     align_pos2[j] = nx.get_node_attributes(G, 'pos')[nodes_G[i]]  # Align pos2 nodes to pos1 nodes

# # Debugging: Check align_pos2 before using it
# print("Aligned Positions in align_pos2:")
# for node, pos in align_pos2.items():
#     print(f"Node {node}: Position {pos}")

# # Ensure that `align_pos2` has positions for all nodes in `G_prior`
# missing_positions = [node for node in G_prior.nodes if node not in align_pos2]
# if missing_positions:
#     print(f"Warning: Missing positions for the following nodes in align_pos2: {missing_positions}")

# # Visualization code
# plt.figure(figsize=(12, 6))

# # Create the subplots first
# ax1 = plt.subplot(1, 2, 1)
# plt.title('Detected Pole Graph (G)')

# # Create the second subplot (aligned graph)
# ax2 = plt.subplot(1, 2, 2)
# plt.title('Aligned Prior Knowledge Graph (G_prior)')

# # Draw the nodes and edges for both graphs
# nx.draw_networkx(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax1, node_size=50, with_labels=False)
# nx.draw_networkx_nodes(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax1, node_size=50)
# nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'), ax=ax1)

# # Draw the aligned nodes and edges for the second graph
# # Ensure that align_pos2 has valid positions before drawing
# if align_pos2:
#     nx.draw_networkx_nodes(G_prior, pos=align_pos2, ax=ax2, node_size=50)
#     nx.draw_networkx_edges(G_prior, pos=align_pos2, ax=ax2)

#     # Add the connections between the matched nodes
#     for i in range(len(nodes_G)):
#         j = np.argmax(X[i]).item()  # Best match in `X`
#         con = ConnectionPatch(
#             xyA=nx.get_node_attributes(G, 'pos')[nodes_G[i]], 
#             xyB=align_pos2[j], 
#             coordsA="data", 
#             coordsB="data",
#             axesA=ax1, 
#             axesB=ax2, 
#             color="green" if X[i, j] else "red",  # Use red for mismatches, green for matches
#             alpha=0.7
#         )
#         plt.gca().add_artist(con)
# else:
#     print("Error: No positions in align_pos2, alignment failed.")

# plt.savefig('../../images/pole_graph_aligned.png')