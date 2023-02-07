import numpy as np
import numpy.testing as npt
import tensornetwork as tn
from .depolarization import depolarization, _get_depolarization_kraus
from ..utils import _split_list, _split_and_alternate


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
        output_edges = depolarization(([rho1[0], rho2[0]], [rho1[1], rho2[1]]), p)

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
