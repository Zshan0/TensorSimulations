import numpy as np
import numpy.testing as npt
import tensornetwork as tn
from .density_gate import z

def test_z():
    nodes_set = set()
    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(np.array([[1.0, 2.0], [3.0, 4.0]]), name="rho")
        rho2 = tn.replicate_nodes([rho1])[0]
        left_edges, right_edges = z(([rho1[0], rho2[0]], [rho1[1], rho2[1]]))

    result = tn.contractors.auto(nodes_set, output_edge_order=left_edges + right_edges)
    one_qubit = np.array([[1, -2], [-3, 4]])
    expected = np.reshape(np.kron(one_qubit, one_qubit), (2, 2, 2, 2))
    npt.assert_array_equal(result.tensor, expected)
