import numpy as np
import tensornetwork as tn

np_vec = np.array([[0], [1]])
tn_vec1 = tn.Node(np_vec)
tn_vec2 = tn.Node(np_vec)

# Quantum gate simulation

qubit = tn.Node(np.array([[0], [1]]))
z1 = tn.Node(np.array([[1, 0], [0, -1]]))
z2 = tn.Node(np.array([[1, 0], [0, -1]]))

# The order of the edge attachment matters
val1 = tn.contract(qubit[0] ^ z1[1])
val2 = tn.contract(z2[1] ^ tn_vec1[0])


def pauli_z(edge):
    z = tn.Node(np.array([[1, 0], [0, -1]]))
    _ = edge ^ z[0]
    return z[1]  # This is the "output edge".


# Build the quantum circuit.


nodes_set = set()
with tn.NodeCollection(nodes_set):
    qubit = tn.Node(np.array([0.0, 1.0]))  # A "zero state" qubit.
    qubit_edge = qubit[0]
    for i in range(1):
        qubit_edge = pauli_z(qubit_edge)

# The result is same in both cases since there is only one edge.
# result = tn.contractors.auto(nodes_set, ignore_edge_order=True)
result = tn.contractors.auto(nodes_set, output_edge_order=[qubit_edge])


# Reusing the network requires replication of nodes.
