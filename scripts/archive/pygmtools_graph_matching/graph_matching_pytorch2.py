# coding: utf-8
"""
===================================================
PyTorch Backend Example: Matching Isomorphic Graphs
===================================================

This example is an introduction to ``pygmtools`` which shows how to match isomorphic graphs.
Isomorphic graphs mean graphs whose structures are identical, but the node correspondence is unknown.
"""

# Author: Runzhong Wang <runzhong.wang@sjtu.edu.cn>
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 6

##############################################################################
# .. note::
#     The following solvers support QAP formulation, and are included in this example:
#
#     * :func:`~pygmtools.classic_solvers.rrwm` (classic solver)
#
#     * :func:`~pygmtools.classic_solvers.ipfp` (classic solver)
#
#     * :func:`~pygmtools.classic_solvers.sm` (classic solver)
#
#     * :func:`~pygmtools.neural_solvers.ngm` (neural network solver)
#
import torch # pytorch backend
import pygmtools as pygm
import matplotlib.pyplot as plt # for plotting
from matplotlib.patches import ConnectionPatch # for plotting matching result
import networkx as nx # for plotting graphs
pygm.set_backend('pytorch') # set default backend for pygmtools
_ = torch.manual_seed(1) # fix random seed

import math
import geojson
from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial import KDTree

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

import networkx as nx
from geopy.distance import geodesic

def geojson_to_cartesian(geojson_file):
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

# Distance threshold
distance_threshold_max = 6.5 /1000 # km
distance_threshold_min = 5.0 /1000 # km

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

##############################################################################
# Generate two isomorphic graphs
# ------------------------------------
#
# num_nodes = 10
# X_gt = torch.zeros(num_nodes, num_nodes)
# X_gt[torch.arange(0, num_nodes, dtype=torch.int64), torch.randperm(num_nodes)] = 1
# A1 = torch.rand(num_nodes, num_nodes)
# A1 = (A1 + A1.t() > 1.) * (A1 + A1.t()) / 2
# torch.diagonal(A1)[:] = 0
# A2 = torch.mm(torch.mm(X_gt.t(), A1), X_gt)
# n1 = torch.tensor([num_nodes])
# n2 = torch.tensor([num_nodes])

geojson_file = '../../data/clustered_poles.geojson'

# Create the graphs
G_ground_truth = create_grid_graph(num_rows, num_cols, row_spacing, col_spacing) # Grount truth graph
G_detection = create_detection_graph(geojson_file) # in long, lat coordinates
# G_detection = geojson_to_cartesian(geojson_file) # in Cartesian coordinates x,y

# Generate adjacency matrices
A1 = nx.adjacency_matrix(G_ground_truth).todense()
A2 = nx.adjacency_matrix(G_detection).todense()

# Convert to PyTorch tensors
A1 = torch.tensor(A1, dtype=torch.float32)
A2 = torch.tensor(A2, dtype=torch.float32)

# Ensure symmetry
A1 = (A1 + A1.T) / 2
A2 = (A2 + A2.T) / 2

# Get the node counts
num_nodes = num_nodes_a = A1.shape[0]
num_nodes_b = A2.shape[0]

# Ground truth matrix X_gt: Here it's a simple identity matrix or you can define it as needed
X_gt = torch.zeros(num_nodes, num_nodes)
X_gt[torch.arange(0, num_nodes, dtype=torch.int64), torch.randperm(num_nodes)] = 1

# Check and fix the node count for A2 if necessary
if num_nodes_b < num_nodes_a:
    # Pad rows (add extra rows in A2 to match the row size of A1)
    padding_b = torch.zeros((num_nodes_a - num_nodes_b, A2.shape[1]))  # Padding rows
    A2 = torch.cat([A2, padding_b], dim=0)  # Add padded rows to A2
    
    # Pad columns (add extra columns in A2 to match the column size of A1)
    padding_b_columns = torch.zeros((A2.shape[0], num_nodes_a - A2.shape[1]))  # Padding columns
    A2 = torch.cat([A2, padding_b_columns], dim=1)  # Add padded columns to A2

# Now both A1 and A2 should have the same number of nodes
n1 = torch.tensor([A1.shape[0]])
n2 = torch.tensor([A2.shape[0]])

print("A1 (ground truth) shape:", A1.shape)  # Should be (num_nodes_b, num_nodes_b)
print("A2 (detected poles)shape:", A2.shape)  # Should be (num_nodes_b, num_nodes_b)


##############################################################################
# Visualize the graphs
# ----------------------
#
plt.figure(figsize=(16, 8))
# G_ground_truth = nx.from_numpy_array(A1.numpy())
# G_detection = nx.from_numpy_array(A2.numpy())

#pos1 = nx.spring_layout(G_ground_truth)
#pos2 = nx.spring_layout(G_detection)

pos1 = nx.get_node_attributes(G_ground_truth, 'pos')
pos2 = nx.get_node_attributes(G_detection, 'pos')

plt.subplot(1, 2, 1)
plt.title('Graph 1 Prior Knowledge')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
plt.subplot(1, 2, 2)

plt.title('Graph 2 Detected Poles')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)
plt.savefig('../../images/graph_matching/pytorch/1_graphs.png')
plt.close()

##############################################################################
# These two graphs look dissimilar because they are not aligned. We then align these two graphs
# by graph matching.
#
# Build affinity matrix
# ----------------------
# To match isomorphic graphs by graph matching, we follow the formulation of Quadratic Assignment Problem (QAP):
#
# .. math::
#
#     &\max_{\mathbf{X}} \ \texttt{vec}(\mathbf{X})^\top \mathbf{K} \texttt{vec}(\mathbf{X})\\
#     s.t. \quad &\mathbf{X} \in \{0, 1\}^{n_1\times n_2}, \ \mathbf{X}\mathbf{1} = \mathbf{1}, \ \mathbf{X}^\top\mathbf{1} \leq \mathbf{1}
#
# where the first step is to build the affinity matrix (:math:`\mathbf{K}`)
#
conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1) # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

# ##############################################################################
# # Visualization of the affinity matrix. For graph matching problem with :math:`N` nodes, the affinity matrix
# # has :math:`N^2\times N^2` elements because there are :math:`N^2` edges in each graph.
# #
# # .. note::
# #     The diagonal elements of the affinity matrix are empty because there is no node features in this example.
# #
# plt.figure(figsize=(4, 4))
# plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
# plt.imshow(K.numpy(), cmap='Blues')
# plt.savefig('../../images/graph_matching/pytorch/2_affinity_matrix.png')
# plt.close()

# ##############################################################################
# # Solve graph matching problem by RRWM solver
# # -------------------------------------------
# # See :func:`~pygmtools.classic_solvers.rrwm` for the API reference.
# #
# X = pygm.rrwm(K, n1, n2)

# ##############################################################################
# # The output of RRWM is a soft matching matrix. Visualization:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('RRWM Soft Matching Matrix')
# plt.imshow(X.numpy(), cmap='Blues')

# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')
# plt.savefig('../../images/graph_matching/pytorch/3_rrwm_soft_matching_matrix.png')
# plt.close()

# ##############################################################################
# # Get the discrete matching matrix
# # ---------------------------------
# # Hungarian algorithm is then adopted to reach a discrete matching matrix
# #
# X = pygm.hungarian(X)

##############################################################################
# NGM neural network solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.ngm` for the API reference.
#
with torch.set_grad_enabled(False):
    X = pygm.ngm(K, n1, n2, pretrain='voc')
    X = pygm.hungarian(X)

##############################################################################
# Visualization of the discrete matching matrix:
#
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
plt.imshow(X.numpy(), cmap='Blues')

plt.subplot(1, 2, 2)
plt.title('Ground Truth Matching Matrix')
plt.imshow(X_gt.numpy(), cmap='Blues')
plt.savefig('../../images/graph_matching/pytorch/4_rrwm_matching_matrix.png')
plt.close()

##############################################################################
# Align the original graphs
# --------------------------
# Draw the matching (green lines for correct matching, red lines for wrong matching):
#
plt.figure(figsize=(16, 8))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1 Prior Knowledge')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2 Detected Poles')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)

number_of_nodes = 0

# for i in range(num_nodes):
#     j = torch.argmax(X[i]).item()
#     con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
#                           axesA=ax1, axesB=ax2, color="green" if X_gt[i, j] else "red")
#     plt.gca().add_artist(con)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes_b:  # Stop if we exceed the number of nodes in G_detection
        break
    j = np.argmax(X[i]).item()  # Find the best match in `X`
    
    # Handle case where j might be out of range in pos2
    if j >= len(pos2):  # Ensure j is within the bounds of pos2
        print(f"Warning: Index {j} out of range for pos2, skipping node {node1}")
        continue
    
    node2 = list(pos2.keys())[j]  # Get the corresponding key in `pos2`

    print(f"Matched node {node1} in G_ground_truth to node {node2} in G_detection")

    # Add a connection line between the two graphs
    con = ConnectionPatch(
        xyA=pos1[node1],  # Get the position of node1
        xyB=pos2[node2],  # Get the position of node2
        coordsA="data", 
        coordsB="data",
        axesA=ax1, 
        axesB=ax2, 
        color="green" if G_detection.has_edge(node1, node2) else "red",
        alpha=0.7
    )
    plt.gca().add_artist(con)
    
    number_of_nodes += 1

print("Number of nodes matched:", number_of_nodes)
    
plt.savefig('../../images/graph_matching/pytorch/5_align_original_graphs.png')
plt.close()

##############################################################################
# Align the nodes:
#
align_A2 = torch.mm(torch.mm(X, A2), X.t())
plt.figure(figsize=(16, 8))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Aligned Graph 2')
align_pos2 = {}

number_of_nodes = 0
missing_nodes = []

# First, initialize positions for all nodes in G_detection with default positions (if needed).
for node in G_detection.nodes():
    if node not in align_pos2:
        align_pos2[node] = pos2.get(node, (0.0, 0.0))  # Default to (0.0, 0.0) if no position is found.

# for i in range(num_nodes):
#     j = torch.argmax(X[i]).item()
#     align_pos2[j] = pos1[i]
#     con = ConnectionPatch(xyA=pos1[i], xyB=align_pos2[j], coordsA="data", coordsB="data",
#                           axesA=ax1, axesB=ax2, color="green" if X_gt[i, j] else "red")
#     plt.gca().add_artist(con)
# nx.draw_networkx(G_detection, pos=align_pos2)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes_b:  # Stop if we exceed the number of nodes in G_detection
        break
    j = np.argmax(X[i]).item()  # Find the best match in `X`
    
    # Handle case where j might be out of range in pos2
    if j >= len(pos2):  # Ensure j is within the bounds of pos2
        print(f"Warning: Index {j} out of range for pos2, skipping node {node1}")
        continue
    
    node2 = list(pos2.keys())[j]  # Get the corresponding key in `pos2`

    print(f"Matched node {node1} in G_ground_truth to node {node2} in G_detection")
    
    # Set the position for node2 in the aligned graph
    align_pos2[node2] = pos1[node1]

    # Add a connection line between the two graphs
    con = ConnectionPatch(
        xyA=pos1[node1],  # Get the position of node1
        xyB=pos2[node2],  # Get the position of node2
        coordsA="data", 
        coordsB="data",
        axesA=ax1, 
        axesB=ax2, 
        color="green" if G_detection.has_edge(node1, node2) else "red",
        alpha=0.7
    )
    plt.gca().add_artist(con)
    
    number_of_nodes += 1

# Identify missing nodes from G_detection compared to G_ground_truth
for node1 in pos1.keys():
    if node1 not in align_pos2:  # If node1 in G_ground_truth is missing in align_pos2 (G_detection)
        missing_nodes.append(node1)
        # Add the missing node from pos1 into align_pos2 at the same position
        pos2[node1] = pos1[node1]  # Assign the position of the missing node from the ground truth
        print(f"Missing node {node1} in G_detection, adding to align_pos2 at position {pos1[node1]}")

print("Number of nodes aligned:", number_of_nodes)
print("Missing nodes in G_detection compared to G_ground_truth:", missing_nodes)

# nx.draw_networkx(G_detection, pos=align_pos2, node_color='green', node_size=100, with_labels=False)
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)
    
plt.savefig('../../images/graph_matching/pytorch/6_align_nodes.png')
plt.close()

# ##############################################################################
# # Other solvers are also available
# # ---------------------------------
# #
# # Classic IPFP solver
# # ^^^^^^^^^^^^^^^^^^^^^
# # See :func:`~pygmtools.classic_solvers.ipfp` for the API reference.
# #
# X = pygm.ipfp(K, n1, n2)

# ##############################################################################
# # Visualization of IPFP matching result:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'IPFP Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X.numpy(), cmap='Blues')

# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')
# plt.savefig('../../images/graph_matching/pytorch/7_ipfp_matching_matrix.png')
# plt.close()

# ##############################################################################
# # Classic SM solver
# # ^^^^^^^^^^^^^^^^^^^^^
# # See :func:`~pygmtools.classic_solvers.sm` for the API reference.
# #
# X = pygm.sm(K, n1, n2)
# X = pygm.hungarian(X)

# ##############################################################################
# # Visualization of SM matching result:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'SM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X.numpy(), cmap='Blues')

# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')
# plt.savefig('../../images/graph_matching/pytorch/8_sm_matching_matrix.png')
# plt.close()

# ##############################################################################
# # NGM neural network solver
# # ^^^^^^^^^^^^^^^^^^^^^^^^^
# # See :func:`~pygmtools.neural_solvers.ngm` for the API reference.
# #
# with torch.set_grad_enabled(False):
#     X = pygm.ngm(K, n1, n2, pretrain='voc')
#     X = pygm.hungarian(X)

# ##############################################################################
# # Visualization of NGM matching result:
# #
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'NGM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X.numpy(), cmap='Blues')

# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')
# plt.savefig('../../images/graph_matching/pytorch/9_ngm_matching_matrix.png')
# plt.close()
