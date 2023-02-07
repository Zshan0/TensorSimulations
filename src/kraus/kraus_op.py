import tensornetwork as tn

import numpy as np
import numpy.testing as npt
import numpy.typing as nptt
import tensornetwork as tn
import cirq


def _get_depolarization_kraus(num_qubits: int, p: float = 0.1):
    return cirq.kraus(cirq.depolarize(p, num_qubits))


def _split_list(a_list):
    half = len(a_list)//2
    return a_list[:half], a_list[half:]


def _split_and_alternate(row:tn.Edge, col:tn.Edge, qubit_count:int) -> list[tn.Edge]:
    """
    Breaks the row, col edges for the nxn Kraus operator to obtain the higher
    dimensional tensor of the shape (2, 2, ...). The function also returns the
    order in traditional tensor representation with interleaving edges.
    """
    dim = tuple([2] * qubit_count)
    exp_row = tn.split_edge(row, dim, [f"row:{ind}" for ind in range(len(dim))])
    exp_col = tn.split_edge(col, dim, [f"col:{ind}" for ind in range(len(dim))])
    # Interleaving row and column to obtain the traditional edge set
    return [x for edge_pair in zip(exp_row, exp_col) for x in edge_pair]


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

        # The first edge is the kraus operator picking edge.
        assert len(self.edges) == 3
        _split_and_alternate(self.edges[1], self.edges[2], num_qubits)


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
) ->  list[tn.Edge]:
    _ = kraus.edges[0] ^ kraus_t.edges[0]

    k_row, k_col = _split_list(kraus.edges[1:])
    kt_row, kt_col = _split_list(kraus_t.edges[1:])

    for (k, k_t, rho_row, rho_col) in zip(k_row, kt_col, rho[0], rho[1]):
        _ = k ^ rho_col
        _ = rho_row ^ k_t

    return kt_row + k_col


def depolarization(rho, p):
    """
    Apply depolarization channel to many-qubit system.

    Input:
        - rho: edges of the many-qubit system. [[n edges], [n edges]]
        - p: Parameter for the depolarization channel

    Returns:
        - Edges for the new density matrix in the standard order of 
            (row, col, rol, ...)
    """
    num_qubits = len(rho[0])
    kraus = Depolarization(num_qubits, p)
    kraus_t = Depolarization(num_qubits, p, is_conj=True)
    return apply_kraus(rho, kraus, kraus_t)


def _get_sum(kraus: tn.Node):
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
    """
    Testing the logic of attaching the kraus picker edge leads to product and 
    then sum.
    """
    np_kraus = np.array(
        [
            [[1, 2], [3, 4]],
            [[-1, -2], [-3, -4]],
        ]
    )

    nodes_set = set()
    with tn.NodeCollection(nodes_set):
        tn_kraus = tn.Node(np_kraus)
        _get_sum(tn_kraus)

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
    # Constructing Kraus for two qubit by taking cartesian product.
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
        output_edges = apply_kraus(
            ([rho1[0], rho2[0]], [rho1[1], rho2[1]]), tn_kraus, tn_kraus_t
        )

    row, col = _split_list(output_edges)
    output_edges = [x for edge_pair in zip(row, col) for x in edge_pair]

    tn_result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)
    result = tn_result.tensor
    # result = np.reshape(tn_result.tensor, (4, 4), 'F')

    # Applying the formula of kraus operators
    expected = np.sum([(k @ rho_2 @ k.conj().T) for k in kraus_two_qubit], axis=0)

    # Constructing the (2, 2, 2, 2) tensor for comparison.
    tn_exp = tn.Node(expected)
    exp_edges = _split_and_alternate(*tn_exp.edges, qubit_count=2)
    reshaped_exp = tn.contractors.auto({tn_exp}, output_edge_order=exp_edges).tensor

    npt.assert_allclose(result, reshaped_exp)

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
        output_edges = apply_kraus(
            ([rho1[0]], [rho1[1]]), tn_kraus, tn_kraus_t
        )

    result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)
    # Applying the formula of krhalf_pointrators
    expected = (kraus_1 @ rho @ kraus_1) + (kraus_2 @ rho @ kraus_2)
    npt.assert_array_equal(result.tensor, expected)

def test_depol_one_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    p = 0.7
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_ops = _get_depolarization_kraus(1, p)

    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(rho, name="rho")
        output_edges = depolarization(([rho1[0]], [rho1[1]]), p)

    result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)
    # Applying the formula of kraus operators
    expected = np.reshape(
        np.sum([kraus @ rho @ kraus.conj().T for kraus in kraus_ops], axis=0),
        (2, 2),
    )
    npt.assert_allclose(result.tensor, expected)

def test_depol_two_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 2.0], [3.0, 4.0]])
    rho_2 = np.kron(rho, rho)

    p = 0.7
    kraus_ops = _get_depolarization_kraus(2, p)

    with tn.NodeCollection(nodes_set):
        rho1 = tn.Node(rho, name="rho")
        rho2 = tn.Node(rho, name="rho")
        output_edges = depolarization(
            ([rho1[0], rho2[0]], [rho1[1], rho2[1]]), p
        )

    row, col = _split_list(output_edges)
    output_edges = [x for edge_pair in zip(row, col) for x in edge_pair]

    tn_result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)
    result = tn_result.tensor
    # Applying the formula of kraus operators
    expected = np.sum([(k @ rho_2 @ k.conj().T) for k in kraus_ops], axis=0)

    tn_exp = tn.Node(expected)
    exp_edges = _split_and_alternate(*tn_exp.edges, qubit_count=2)
    reshaped_exp = tn.contractors.auto({tn_exp}, output_edge_order=exp_edges).tensor

    npt.assert_allclose(result, reshaped_exp)
