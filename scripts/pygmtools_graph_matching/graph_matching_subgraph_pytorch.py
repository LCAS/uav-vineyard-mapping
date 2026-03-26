# coding: utf-8
"""
==============================================
PyTorch Backend Example: Discovering Subgraphs
==============================================

This example shows how to match a smaller graph to a subset of a larger graph.
"""

# Author: Runzhong Wang <runzhong.wang@sjtu.edu.cn>
#
# License: Mulan PSL v2 License
# sphinx_gallery_thumbnail_number = 5

##############################################################################
# .. note::
#     The following solvers are included in this example:
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
import numpy as np

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

# ##############################################################################
# # Generate the larger graph
# # --------------------------
# #
# num_nodes2 = 10
# A2 = torch.rand(num_nodes2, num_nodes2)
# A2 = (A2 + A2.t() > 1.) * (A2 + A2.t()) / 2
# torch.diagonal(A2)[:] = 0
# n2 = torch.tensor([num_nodes2])

# ##############################################################################
# # Generate the smaller graph
# # ---------------------------
# #
# num_nodes1 = 5
# G2 = nx.from_numpy_array(A2.numpy())
# pos2 = nx.spring_layout(G2)
# pos2_t = torch.tensor([pos2[_] for _ in range(num_nodes2)])
# selected = [0] # build G1 as a cluster in visualization
# unselected = list(range(1, num_nodes2))
# while len(selected) < num_nodes1:
#     dist = torch.sum(torch.sum(torch.abs(pos2_t[selected].unsqueeze(1) - pos2_t[unselected].unsqueeze(0)), dim=-1), dim=0)
#     select_id = unselected[torch.argmin(dist).item()] # find the closest node from unselected
#     selected.append(select_id)
#     unselected.remove(select_id)
# selected.sort()
# A1 = A2[selected, :][:, selected]
# X_gt = torch.eye(num_nodes2)[selected, :]
# n1 = torch.tensor([num_nodes1])

# Generate the graphs

geojson_file = '../../data/clustered_poles.geojson'

# Create the graphs
G_ground_truth = create_grid_graph(num_rows, num_cols, row_spacing, col_spacing) # Grount truth graph
G_detection = create_detection_graph(geojson_file) # in long, lat coordinates

# Generate adjacency matrices
A1 = nx.adjacency_matrix(G_ground_truth).todense()
A2 = nx.adjacency_matrix(G_detection).todense()

# Convert to PyTorch tensors
A1 = torch.tensor(A1, dtype=torch.float32)
A2 = torch.tensor(A2, dtype=torch.float32)

n1 = torch.tensor([A1.shape[0]])
n2 = torch.tensor([A2.shape[0]])

num_nodes1 = A1.shape[0]
num_nodes2 = A2.shape[0]

# Ground truth matrix X_gt: Here it's a simple identity matrix or you can define it as needed
X_gt = torch.zeros(A1.shape[0], A1.shape[0])
X_gt[torch.arange(0, A1.shape[0], dtype=torch.int64), torch.randperm(A1.shape[0])] = 1

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
plt.savefig('../../images/graph_matching/subgraph/pytorch/1_graphs.png')
plt.close()

##############################################################################
# We then show how to automatically discover the matching by graph matching.
#
# Build affinity matrix
# ----------------------
# To match the larger graph and the smaller graph, we follow the formulation of Quadratic Assignment Problem (QAP):
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
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.001) # set affinity function
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

##############################################################################
# Visualization of the affinity matrix. For graph matching problem with :math:`N_1` and :math:`N_2` nodes,
# the affinity matrix has :math:`N_1N_2\times N_1N_2` elements because there are :math:`N_1^2` and
# :math:`N_2^2` edges in each graph, respectively.
#
# .. note::
#     The diagonal elements of the affinity matrix is empty because there is no node features in this example.
#
# plt.figure(figsize=(16, 8))
# plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
# plt.imshow(K.numpy(), cmap='Blues')

##############################################################################
# Solve graph matching problem by RRWM solver
# -------------------------------------------
# See :func:`~pygmtools.classic_solvers.rrwm` for the API reference.
#
X = pygm.rrwm(K, n1, n2)

##############################################################################
# The output of RRWM is a soft matching matrix. Visualization:
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.title('RRWM Soft Matching Matrix')
# plt.imshow(X.numpy(), cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')

# nx.draw_networkx(G2, pos=pos2, node_color=color2)

##############################################################################
# Get the discrete matching matrix
# ---------------------------------
# Hungarian algorithm is then adopted to reach a discrete matching matrix
#
X = pygm.hungarian(X)

##############################################################################
# Visualization of the discrete matching matrix:
#
# plt.figure(figsize=(16, 8))
# plt.subplot(1, 2, 1)
# plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X.numpy(), cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt.numpy(), cmap='Blues')

#############################################################################
# Match the subgraph
# -------------------
# Draw the matching:
#
plt.figure(figsize=(16, 8))
plt.suptitle(f'RRWM Matching Result')
ax1 = plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes2:  # Stop if we exceed the number of nodes in G_detection
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
    

plt.savefig('../../images/graph_matching/subgraph/pytorch/2_RRWM_graph_matching.png')
plt.close()

##############################################################################
# Other solvers are also available
# ---------------------------------
#
# Classic IPFP solver
# ^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.classic_solvers.ipfp` for the API reference.
#
X = pygm.ipfp(K, n1, n2)

##############################################################################
# Visualization of IPFP matching result:
#
plt.figure(figsize=(16, 8))
# plt.suptitle(f'IPFP Matching Result (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
plt.suptitle(f'IPFP Matching Result')
ax1 = plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes2:  # Stop if we exceed the number of nodes in G_detection
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

plt.savefig('../../images/graph_matching/subgraph/pytorch/3_IPFP_graph_matching.png')
plt.close()

##############################################################################
# Classic SM solver
# ^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.classic_solvers.sm` for the API reference.
#
X = pygm.sm(K, n1, n2)
X = pygm.hungarian(X)

##############################################################################
# Visualization of SM matching result:
#
plt.figure(figsize=(16, 8))
# plt.suptitle(f'SM Matching Result (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
plt.suptitle(f'SM Matching Result')
ax1 = plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes2:  # Stop if we exceed the number of nodes in G_detection
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

plt.savefig('../../images/graph_matching/subgraph/pytorch/4_SM_graph_matching.png')
plt.close()

##############################################################################
# NGM neural network solver
# ^^^^^^^^^^^^^^^^^^^^^^^^^
# See :func:`~pygmtools.neural_solvers.ngm` for the API reference.
#
# .. note::
#     The NGM solvers are pretrained on a different problem setting, so their performance may seem inferior.
#     To improve their performance, you may change the way of building affinity matrices, or try finetuning
#     NGM on the new problem.
#
with torch.set_grad_enabled(False):
    X = pygm.ngm(K, n1, n2, pretrain='voc')
    X = pygm.hungarian(X)

##############################################################################
# Visualization of NGM matching result:
#
plt.figure(figsize=(16, 8))
# plt.suptitle(f'NGM Matching Result (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
plt.suptitle(f'NGM Matching Result')
ax1 = plt.subplot(1, 2, 1)
plt.title('Subgraph 1')
nx.draw_networkx(G_ground_truth, pos=pos1, node_color='blue', node_size=100, with_labels=False)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G_detection, pos=pos2, node_color='green', node_size=100, with_labels=False)

# Ensure we are not going out of bounds for G_detection nodes
for i, node1 in enumerate(pos1.keys()):
    if i >= num_nodes2:  # Stop if we exceed the number of nodes in G_detection
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

plt.savefig('../../images/graph_matching/subgraph/pytorch/5_NGM_graph_matching.png')
plt.close()