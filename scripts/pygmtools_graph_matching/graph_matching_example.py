import numpy as np  # numpy backend
import pygmtools as pygm
import matplotlib.pyplot as plt  # for plotting
from matplotlib.patches import ConnectionPatch  # for plotting matching result
import networkx as nx  # for plotting graphs

pygm.set_backend('numpy')  # set default backend for pygmtools
np.random.seed(1)  # fix random seed

output_folder = "../../images/"
image_counter = 1  # Counter for filenames

def save_figure(description):
    """
    Saves the current figure with an incremented numbered filename.
    """
    global image_counter
    filename = f"{output_folder}{image_counter}_{description}.png"
    plt.savefig(filename)
    plt.close()
    image_counter += 1

# Number of nodes
num_nodes = 10

# Generate ground truth permutation matrix
X_gt = np.zeros((num_nodes, num_nodes))
X_gt[np.arange(0, num_nodes, dtype=np.int64), np.random.permutation(num_nodes)] = 1

# Generate adjacency matrices
A1 = np.random.rand(num_nodes, num_nodes)
A1 = (A1 + A1.T > 1.0) * (A1 + A1.T) / 2
np.fill_diagonal(A1, 0)
A2 = np.matmul(np.matmul(X_gt.T, A1), X_gt)
n1 = np.array([num_nodes])
n2 = np.array([num_nodes])

# Plot the graphs
plt.figure(figsize=(8, 4))
G1 = nx.from_numpy_array(A1)
G2 = nx.from_numpy_array(A2)
pos1 = nx.spring_layout(G1)
pos2 = nx.spring_layout(G2)
plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2)
save_figure("graph_comparison")

# Build affinity matrix
conn1, edge1 = pygm.utils.dense_to_sparse(A1)
conn2, edge2 = pygm.utils.dense_to_sparse(A2)
import functools
gaussian_aff = functools.partial(pygm.utils.gaussian_aff_fn, sigma=.1)
K = pygm.utils.build_aff_mat(None, edge1, conn1, None, edge2, conn2, n1, None, n2, None, edge_aff_fn=gaussian_aff)

# # Plot affinity matrix
# plt.figure(figsize=(4, 4))
# plt.title(f'Affinity Matrix (size: {K.shape[0]}$\\times${K.shape[1]})')
# plt.imshow(K, cmap='Blues')
# save_figure("affinity_matrix")

# RRWM solver
X = pygm.rrwm(K, n1, n2)

# # Plot RRWM results
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title('RRWM Soft Matching Matrix')
# plt.imshow(X, cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt, cmap='Blues')
# save_figure("rrwm_soft_matching")

# Hungarian solver for discrete matching
X = pygm.hungarian(X)

# # Plot Hungarian results
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'RRWM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X, cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt, cmap='Blues')
# save_figure("rrwm_matching")

# Graph matching visualization
plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
ax2 = plt.subplot(1, 2, 2)
plt.title('Graph 2')
nx.draw_networkx(G2, pos=pos2)
for i in range(num_nodes):
    j = np.argmax(X[i]).item()
    con = ConnectionPatch(xyA=pos1[i], xyB=pos2[j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="green" if X_gt[i, j] else "red")
    plt.gca().add_artist(con)
save_figure("graph_matching")

# Align graph 2 to graph 1
align_A2 = np.matmul(np.matmul(X, A2), X.T)
plt.figure(figsize=(8, 4))
ax1 = plt.subplot(1, 2, 1)
plt.title('Graph 1')
nx.draw_networkx(G1, pos=pos1)
ax2 = plt.subplot(1, 2, 2)
plt.title('Aligned Graph 2')
align_pos2 = {}
for i in range(num_nodes):
    j = np.argmax(X[i]).item()
    align_pos2[j] = pos1[i]
    con = ConnectionPatch(xyA=pos1[i], xyB=align_pos2[j], coordsA="data", coordsB="data",
                          axesA=ax1, axesB=ax2, color="green" if X_gt[i, j] else "red")
    plt.gca().add_artist(con)
nx.draw_networkx(G2, pos=align_pos2)
save_figure("aligned_graph")

# # IPFP solver
# X = pygm.ipfp(K, n1, n2)

# # Plot IPFP results
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'IPFP Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X, cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt, cmap='Blues')
# save_figure("ipfp_matching")

# # SM solver
# X = pygm.sm(K, n1, n2)
# X = pygm.hungarian(X)

# # Plot SM results
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'SM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X, cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt, cmap='Blues')
# save_figure("sm_matching")

# # NGM solver
# X = pygm.ngm(K, n1, n2, pretrain='voc')
# X = pygm.hungarian(X)

# # Plot NGM results
# plt.figure(figsize=(8, 4))
# plt.subplot(1, 2, 1)
# plt.title(f'NGM Matching Matrix (acc={(X * X_gt).sum()/ X_gt.sum():.2f})')
# plt.imshow(X, cmap='Blues')
# plt.subplot(1, 2, 2)
# plt.title('Ground Truth Matching Matrix')
# plt.imshow(X_gt, cmap='Blues')
# save_figure("ngm_matching")
