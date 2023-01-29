import numpy as np
import numpy.testing as npt
import tensornetwork as tn

# Quantum gate simulation via density operator.
class Z(tn.Node):
    def __init__(self, is_conj=False):
        tenz = np.array([[1, 0], [0, -1]])
        if not is_conj:
            super().__init__(tenz, name="PauliZ")
        else:
            super().__init__(tenz.conj().T, name="PauliZ")


def rho_z(rho):
    # Assuming the application is only on one qubit.
    z = Z()
    z_t = Z(True)
    _ = z[0] ^ rho[1]
    _ = rho[0] ^ z_t[1]
    return [z_t[0], z[1]]


def many_z(rho):
    """
    Apply z gate to n qubit system.

    Input:
        - rho: edges of the two qubit system. [[n edges], [n edges]]

    Returns:
        - Edges for the new density matrix.
    """
    zs = [Z() for _ in rho[0]]

    zs_t = [Z(True) for _ in rho[0]]

    for (z, z_t, e1, e2) in zip(zs, zs_t, rho[0], rho[1]):
        _ = z[0] ^ e1
        _ = e2 ^ z_t[1]
    return ([z[1] for z in zs], [z_t[0] for z_t in zs_t])


def test_z():
    nodes_set = set()
    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(np.array([[1.0, 2.0], [3.0, 4.0]]), name="rho")
        rho2 = tn.replicate_nodes([rho1])[0]
        left_edges, right_edges = many_z(([rho1[0], rho2[0]], [rho1[1], rho2[1]]))

    # result = tn.contractors.auto(nodes_set, ignore_edge_order=True)
    result = tn.contractors.auto(nodes_set, output_edge_order=left_edges + right_edges)
    one_qubit = np.array([[1, -2], [-3, 4]])
    expected = np.reshape(np.kron(one_qubit, one_qubit), (2, 2, 2, 2))
    npt.assert_array_equal(result.tensor, expected)
