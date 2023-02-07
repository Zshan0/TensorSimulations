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

