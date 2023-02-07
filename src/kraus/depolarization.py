import tensornetwork as tn
import cirq
from .kraus import Kraus, apply_kraus


def _get_depolarization_kraus(num_qubits: int, p: float = 0.1):
    return cirq.kraus(cirq.depolarize(p, num_qubits))


def _split_and_alternate(row: tn.Edge, col: tn.Edge, qubit_count: int) -> list[tn.Edge]:
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


class Depolarization(Kraus):
    def __init__(self, num_qubits: int, p: float, is_conj=False, label=""):
        """
        Get depolarization channel for :math:`n` qubits. The tensor will be a
        list of :math:`4^n` operators of shape :math:`(2, 2, ...)`, a series
        of twos of length :math:`2n`.
        """
        ops = _get_depolarization_kraus(num_qubits, p)
        super().__init__(ops, is_conj=is_conj, label="depol" + label)


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
