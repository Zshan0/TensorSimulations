import tensornetwork as tn

import numpy as np
import numpy.testing as npt
import tensornetwork as tn
import cirq


def _get_depolarization_kraus(num_qubits: int, p: float = 0.1):
    return cirq.kraus(cirq.depolarize(p, num_qubits))

class Kraus(tn.Node):
    def __init__(self, ops, is_conj=False, label=""):
        """
        Construct the Node representing the Kraus operators. The tensor will be 
        a list of :math:`4^n` operators of shape :math:`(2, 2, ...)`, a series
        of twos of length :math:`2n`.
        """
        if not is_conj:
            super().__init__(np.array(ops), name="kraus_" + label)
        else:
            ops = [x.conj().T for x in ops]
            super().__init__(np.array(ops), name="kraus_t" + label)
        self._reshape_qubit()

    def _reshape_qubit(self):
        """
        Reshape (k, n, n) into (k, 2, 2, ...) to break the individual kraus
        operator's edges into qubit edges.
        """
        num_qubits = int(np.log2(self.edges[1].dimension))
        dim = tuple([2] * num_qubits)

        # The first edge is the kraus operator picking edge.
        for edge in self.edges[1:]:
            tn.split_edge(edge, dim)

        assert False


class Depolarization(Kraus):
    def __init__(self, num_qubits: int, p: float, is_conj=False, label=""):
        """
        Get depolarization channel for :math:`n` qubits. The tensor will be a
        list of :math:`4^n` operators of shape :math:`(2, 2, ...)`, a series
        of twos of length :math:`2n`.
        """
        ops = _get_depolarization_kraus(num_qubits, p)
        super().__init__(ops, is_conj=is_conj, label="depol" + label)



def apply_kraus(
    rho: tuple[list[tn.Edge], list[tn.Edge]], kraus: tn.Node, kraus_t: tn.Node
):
    _ = kraus.edges[0] ^ kraus_t.edges[0]

    # We need to skip the first one because that is on the Kraus operators.
    # That leaves alternate ones row and column of the nested tensor.
    k_row = kraus.edges[1::2]
    k_col = kraus.edges[2::2]

    kt_row = kraus_t.edges[1::2]
    kt_col = kraus_t.edges[2::2]

    for (k, k_t, rho_row, rho_col) in zip(k_row, kt_col, rho[0], rho[1]):
        _ = k ^ rho_col
        _ = rho_row ^ k_t

    return (kt_row, k_col)


def depolarization(rho, p):
    """
    Apply depolarization channel to many-qubit system.

    Input:
        - rho: edges of the many-qubit system. [[n edges], [n edges]]
        - p: Parameter for the depolarization channel

    Returns:
        - Edges for the new density matrix.
    """
    num_qubits = len(rho[0])
    kraus = Depolarization(num_qubits, p)
    kraus_t = Depolarization(num_qubits, p, is_conj=True)
    return apply_kraus(rho, kraus, kraus_t)


def get_sum(kraus: tn.Node):
    """
    We construct a identity matrix that is compatible with the kraus operator
    and contract the edges in such a way that We get :math:`\\Sum_i K_i x I`.
    """
    shape = kraus.shape

    eye = np.tile(np.identity(2), shape[:-2])
    eye = np.reshape(eye.T, shape)
    tn_eye = tn.Node(eye)
    _ = kraus.edges[0] ^ tn_eye.edges[0]
    _ = kraus.edges[1] ^ tn_eye.edges[2]


def test_sum():
    np_kraus = np.array(
        [
            [[1, 2], [3, 4]],
            [[-1, -2], [-3, -4]],
        ]
    )

    nodes_set = set()
    with tn.NodeCollection(nodes_set):
        tn_kraus = tn.Node(np_kraus)
        get_sum(tn_kraus)

    result = tn.contractors.auto(nodes_set, ignore_edge_order=True)
    # Sum of the matrices comes to 0.
    expected = np.zeros((2, 2))
    npt.assert_array_equal(result.tensor, expected)


def test_kraus_two_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_2 = np.kron(rho, rho)
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_1 = np.array([[0.9, 0.0], 
                        [0.0, 0.9]])

    kraus_2 = np.array([[0.0, 0.1], 
                        [0.1, 0.0]])
    kraus_one_qubit = [kraus_1, kraus_2]
    kraus_two_qubit = tuple([
            np.kron(x, y)
            for x in kraus_one_qubit
            for y in kraus_one_qubit
        ])

    with tn.NodeCollection(nodes_set):
        tn_kraus = Kraus(kraus_two_qubit)
        # tn_kraus.

        # Hermitian operators
        tn_kraus_t = Kraus(kraus_two_qubit, is_conj=True)

        rho1 = tn.Node(rho, name="rho1")
        rho2 = tn.Node(rho, name="rho2")
        apply_kraus(
            ([rho1[0], rho2[0]], [rho1[1], rho2[1]]), tn_kraus, tn_kraus_t
        )
        # left_edges, right_edges = apply_kraus(
        #     ([rho1[0], rho2[0]], [rho1[1], rho2[1]]), tn_kraus, tn_kraus_t
        # )

    result = tn.contractors.auto(nodes_set, ignore_edge_order=True)

    # Applying the formula of kraus operators
    expected = np.sum([(k @ rho_2 @ k.conj().T) for k in kraus_two_qubit], axis=0)

    # e0, e1, e2, e3 = result.edges
    # tn.flatten_edges([e0, e2])
    # tn.flatten_edges([e1, e3])
    npt.assert_allclose(result.tensor, expected)

def test_kraus_one_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_1 = np.array([[0.9, 0.0], [0.0, 0.9]])
    kraus_2 = np.array([[0.0, 0.1], [0.1, 0.0]])

    with tn.NodeCollection(nodes_set):
        tn_kraus = tn.Node(np.array([kraus_1, kraus_2]), name="Kraus")

        # Hermitian operators
        tn_kraus_t = tn.Node(np.array([kraus_1, kraus_2]), name="Kraus_t")

        rho1 = tn.Node(rho, name="rho")
        left_edges, right_edges = apply_kraus(
            ([rho1[0]], [rho1[1]]), tn_kraus, tn_kraus_t
        )

    result = tn.contractors.auto(nodes_set, output_edge_order=left_edges + right_edges)
    # Applying the formula of kraus operators
    expected = (kraus_1 @ rho @ kraus_1) + (kraus_2 @ rho @ kraus_2)
    npt.assert_array_equal(result.tensor, expected)


def test_val():
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    rho2 = np.kron(rho, rho)
    rho3 = np.array([rho2, rho2])
    tn_rho = tn.Node(rho3)
    e0, e1 = tn_rho.edges[1:]
    assert False
    tn.split_edge(e0, (2, 2))
    tn.split_edge(e1, (2, 2))


def test_depol_one_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = 0.7
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_ops = _get_depolarization_kraus(1, p)

    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(rho, name="rho")
        left_edges, right_edges = depolarization(([rho1[0]], [rho1[1]]), p)

    result = tn.contractors.auto(nodes_set, output_edge_order=left_edges + right_edges)
    # Applying the formula of kraus operators
    expected = np.reshape(
        np.sum([kraus @ rho @ kraus.conj().T for kraus in kraus_ops], axis=0),
        (2, 2),
    )
    npt.assert_allclose(result.tensor, expected)

def test_depol_two_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = 0.7
    kraus_ops = _get_depolarization_kraus(2, p)

    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(rho, name="rho")
        rho2 = tn.Node(rho, name="rho")
        left_edges, right_edges = depolarization(
            ([rho1[0], rho2[0]], [rho1[1], rho2[1]]), p
        )

    result = tn.contractors.auto(nodes_set, output_edge_order=left_edges + right_edges)
    # Applying the formula of kraus operators
    rho_two_qubit = np.kron(rho, rho)
    expected = np.reshape(
        np.sum([kraus @ rho_two_qubit @ kraus.conj().T for kraus in kraus_ops], axis=0),
        (2, 2, 2, 2),
    )
    npt.assert_array_equal(result.tensor, expected)
