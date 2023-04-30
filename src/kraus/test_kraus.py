import numpy as np
import numpy.testing as npt
import tensornetwork as tn
from .kraus import Kraus, apply_kraus
from ..utils import _split_list, _split_and_alternate


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
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    rho_2 = np.kron(rho, rho)

    kraus_1 = np.array([[0.9, 0.3], [3.0, 0.9]])
    kraus_2 = np.array([[0.2, 0.1], [0.1, 0.0]])
    kraus_one_qubit = np.array([kraus_1, kraus_2])

    # Constructing Kraus for two qubit by taking cartesian product.
    kraus_two_qubit = tuple(
        [np.kron(x, y) for x in kraus_one_qubit for y in kraus_one_qubit]
    )

    nodes_set = set()
    with tn.NodeCollection(nodes_set):
        tn_kraus = Kraus(kraus_two_qubit)
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

    # Applying the formula of kraus operators
    expected = np.sum([(k @ rho_2 @ k.conj().T) for k in kraus_two_qubit], axis=0)

    # Constructing the (2, 2, 2, 2) tensor for comparison.
    tn_exp = tn.Node(expected)
    exp_edges = _split_and_alternate(*tn_exp.edges, qubit_count=2)
    reshaped_exp = tn.contractors.auto({tn_exp}, output_edge_order=exp_edges).tensor

    npt.assert_allclose(result, reshaped_exp)


# Testing the logic for Kraus operators
def test_kraus_one_qubit_logic():
    nodes_set = set()
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_1 = np.array([[2.0, 0.0], [0.0, 2.0]])
    kraus_2 = np.array([[1.0, 0.0], [1.0, 0.0]])

    kraus_ops = np.array([kraus_1, kraus_2])
    kraus_ops_t = np.array([kraus_1.conj().T, kraus_2.conj().T])

    with tn.NodeCollection(nodes_set):
        tn_kraus = tn.Node(kraus_ops, name="Kraus")
        tn_kraus_t = tn.Node(kraus_ops_t, name="Kraus_t")

        rho1 = tn.Node(rho, name="rho")

    # Connecting the sum loop
    _ = tn_kraus[0] ^ tn_kraus_t[0]

    # Connecting the row, col edges
    _ = tn_kraus[2] ^ rho1[0]
    _ = tn_kraus_t[1] ^ rho1[1]

    output_edges = [tn_kraus[1], tn_kraus_t[2]]
    result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)

    expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    for kraus, kraus_t in zip(kraus_ops, kraus_ops_t):
        expected += kraus @ rho @ kraus_t
    npt.assert_array_equal(result.tensor, expected)


def test_kraus_one_qubit():
    nodes_set = set()
    rho = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Kraus ops corresponding to bitflip error of 0.1 chance
    kraus_1 = np.array([[2.0, 0.0], [0.0, 2.0]])
    kraus_2 = np.array([[1.0, 0.0], [1.0, 0.0]])

    kraus_ops = np.array([kraus_1, kraus_2])
    kraus_ops_t = np.array([kraus_1.conj().T, kraus_2.conj().T])

    with tn.NodeCollection(nodes_set):
        tn_kraus = tn.Node(kraus_ops, name="Kraus")
        tn_kraus_t = tn.Node(kraus_ops_t, name="Kraus_t")
        rho1 = tn.Node(rho, name="rho")

    output_edges = apply_kraus(([rho1[0]], [rho1[1]]), tn_kraus, tn_kraus_t)
    result = tn.contractors.auto(nodes_set, output_edge_order=output_edges)

    expected = np.array([[0.0, 0.0], [0.0, 0.0]])
    for kraus, kraus_t in zip(kraus_ops, kraus_ops_t):
        expected += kraus @ rho @ kraus_t
    npt.assert_array_equal(result.tensor, expected)


def test_add_multiple():
    A = np.array([[2.0, 0.0], [0.0, 2.0]])
    B = np.array([[1.0, 0.0], [1.0, 0.0]])

    C = np.array([[3.0, 0.0], [4.0, 0.0]])

    T = np.array([A, B])
    Tn = tn.Node(T)
    Cn = tn.Node(C)

    nnodes = tn.replicate_nodes([Tn, Cn])
    # _ = Tn[2] ^ Cn[0]
    _ = nnodes[0][2] ^ nnodes[1][0]
    result = tn.contractors.auto(
        nnodes, output_edge_order=[nnodes[0][0], nnodes[0][1], nnodes[1][1]]
    )

    expected = np.array([A @ C, B @ C])
    npt.assert_array_equal(result.tensor, expected)

    # Checking reverse
    _ = Tn[1] ^ Cn[1]
    result = tn.contractors.auto([Tn, Cn], output_edge_order=[Tn[0], Cn[0], Tn[2]])

    expected = np.array([C @ A, C @ B])
    npt.assert_array_equal(result.tensor, expected)


def test_check_unitary():
    UA = np.array([[1.0, 0.0], [2.0, 3.0]])

    UB = np.array([[3.0, 4.0], [5.0, 6.0]])

    rho = np.array([[1.0, 2.0], [-1.0, -2.0]])

    UAn = tn.Node(UA)
    UBn = tn.Node(UB)
    rhon = tn.Node(rho)

    _ = UAn[1] ^ rhon[0]
    _ = rhon[1] ^ UBn[0]

    result = tn.contractors.auto([UAn, UBn, rhon], output_edge_order=[UAn[0], UBn[1]])

    expected = UA @ rho @ UB
    npt.assert_array_equal(result.tensor, expected)
