import numpy as np
import tensornetwork as tn

np_vec = np.array([[0], [1]])
tn_vec1 = tn.Node(np_vec)
tn_vec2 = tn.Node(np_vec)

# Contracting the first index gives matrix with 1 element
tn_outer = tn.contract(tn_vec1[0] ^ tn_vec2[0])

# Contracting the second index gives matrix
tn_outer1 = tn.contract(tn_vec1[1] ^ tn_vec2[1])
tn_outer2 = tn.contract(tn_vec2[1] ^ tn_vec1[1])

# Matrix multiplication
np_mat1 = np.array([[1, 2], [3, 4]])

np_mat2 = np.array([[0, 1], [1, 0]])

tn_mat1 = tn.Node(np_mat1)
tn_mat2 = tn.Node(np_mat2)

# Multiplying by contracting the edges

mat12 = tn.contract(tn_mat1[1] ^ tn_mat2[0])

# Vector multiplication

vec1 = tn.contract(mat12[0] ^ tn_vec1[0])  # Picks 2nd row.
vec2 = tn.contract(mat12[1] ^ tn_vec1[0])  # Picks 2nd column.


# Quantum gate simulation


qubit = tn.Node(np.array([[0], [1]]))
z = tn.Node(np.array([[1, 0], [0, -1]]))
# _ = z[1] ^ qubit[0]
val = tn.contract(qubit[0] ^ z[1])
# val = qubit @ z
