import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt

# Example graphs
G1 = nx.Graph()
G1.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

G2 = nx.Graph()
positions = {0: (1, 2), 1: (2, 3), 2: (3, 1), 3: (1, 1)}
G2.add_nodes_from(positions.keys())
nx.set_node_attributes(G2, positions, 'pos')

# Extract positions from G2
pos = nx.get_node_attributes(G2, 'pos')
print("Positions in G2:", pos)

# Create a cost matrix based on distances
cost_matrix = np.zeros((len(G1.nodes), len(G2.nodes)))
for i, node1 in enumerate(G1.nodes):
    for j, node2 in enumerate(G2.nodes):
        cost_matrix[i, j] = np.linalg.norm(np.array(pos[node2]) - np.array(pos[node1]))

print("Cost matrix:\n", cost_matrix)

# Solve the assignment problem
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# Create a mapping from G1 to G2
mapping = {list(G1.nodes)[i]: list(G2.nodes)[j] for i, j in zip(row_ind, col_ind)}

print("Node mapping from G1 to G2:", mapping)

# Visualize the graphs and the mapping
plt.figure(figsize=(15, 5))

# Plot G1
plt.subplot(131)
nx.draw(G1, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500)
plt.title("Graph G1")
# plt.savefig("graph_G1.png")

# Plot G2 with positions
plt.subplot(132)
nx.draw(G2, pos=pos, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500)
plt.title("Graph G2")
# plt.savefig("graph_G2.png")

# Plot the mapping between G1 and G2
plt.subplot(133)
combined_graph = nx.Graph()
combined_graph.add_nodes_from(G1.nodes(data=True))
combined_graph.add_nodes_from(G2.nodes(data=True))
combined_graph.add_edges_from(G1.edges(data=True))
combined_graph.add_edges_from(G2.edges(data=True))

# Add edges representing the mapping
for g1_node, g2_node in mapping.items():
    combined_graph.add_edge(g1_node, g2_node)

pos_combined = {**nx.spring_layout(G1), **pos}
nx.draw(combined_graph, pos=pos_combined, with_labels=True, node_color='lightcoral', edge_color='gray', node_size=500)
plt.title("Combined Graph with Mapping")
plt.savefig("graph_combined.png")

print("The graphs and their mapping have been visualized and saved as 'graph_G1.png', 'graph_G2.png', and 'graph_combined.png'.")