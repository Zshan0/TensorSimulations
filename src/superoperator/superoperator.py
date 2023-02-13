import tensornetwork as tn

import numpy as np
import tensornetwork as tn
from ..utils import _split_list, _split_and_alternate
import cirq
import numpy.testing as npt


def flatten_density(rho_edges:tuple[list[tn.Edge], list[tn.Edge]]) -> tn.Edge:
    """
    Flatten the density operator in a vector of dim :math:`d^2 x 1`. The 
    order of the edges will be (c0, c1... r0, r1...rn).

    Input:
        - rho: Density operator edges of form [[row], [col]].
    Return:
        - The edge corresponding to flattened rho.
    """
    row, col = rho_edges
    np_eye = np.eye(len(row) + len(col))
    tn_eye = tn.Node(np_eye)
    row_eye, col_eye = _split_list(tn_eye.edges)
    for r_eye, edge in zip(row_eye, row + col):
        _ = edge ^ r_eye
    return tn.flatten_edges(col_eye, "flattened_density")

def unflatten_density(rho_edge: tn.Edge) -> list[tn.Edge]:
    """
    Unflatten the density operator edge to the shape :math:`(2, 2, ...)`.
    """
    num_qubits = int(np.log2(rho_edge.dimension))
    d = 2 ** num_qubits
    row, col = tn.split_edge(rho_edge, (d, d), ["row", "col"])
    return _split_and_alternate(row, col, num_qubits)

class Superoperator(tn.Node):
    def __init__(self, ops, is_conj=False, label=""):
        """
        Construct the superoperator corresponding to the tensor given. The
        superoperator tensor will be :math:`d^2 x d^2`. Which will be reshaped
        Construct the Node representing the Kraus operators. The tensor will be
        a list of :math:`4^n` operators of shape :math:`(2, 2, ...)`, a series
        of twos of length :math:`2n`.
        """
        if not is_conj:
            super().__init__(np.array(ops), name="kraus_" + label)
        else:
            ops = [x.conj().T for x in ops]
            super().__init__(np.array(ops), name="kraus_t" + label)

def apply_superoperator(rho: tuple[list[tn.Edge], list[tn.Edge]], superoperator: tn.Node) -> list[tn.Edge]:
    """
    Apply the given superoperator node to the density matrix.

    Input:
        - rho: pair of sequences of tuples that take in rows and columns of
               edges.
        - superoperator: Node representing the superoperator.
    """
    flattened_rho = flatten_density(rho)
    _ = superoperator[0] ^ flattened_rho
    return unflatten_density(superoperator[1])


def test_flatten_one_qubit():
    rho = np.array([[1, 2], [3, 4]])
    tn_rho = tn.Node(rho)
    rho_edges = ([tn_rho[0]], [tn_rho[1]])
    _ = flatten_density(rho_edges)
    expected = np.array([1, 2, 3, 4])
    npt.assert_array_equal(tn_rho.tensor, expected)

def test_flatten_two_qubit():
    rho = np.array([[1, 2], [3, 4]])
    tn_rho1 = tn.Node(rho)
    tn_rho2 = tn.Node(rho)
    rho_edges = ([tn_rho1[0], tn_rho2[0]], [tn_rho1[1], tn_rho2[1]])
    _ = flatten_density(rho_edges)
    expected = np.array([1, 2, 3, 4])
    np_rho = np.kron(tn_rho1.tensor, tn_rho2.tensor)
    npt.assert_array_equal(np_rho, expected)
