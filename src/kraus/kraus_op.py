import tensornetwork as tn

import numpy as np
import numpy.typing as npt
import tensornetwork as tn
import cirq


def _get_depolarization_kraus(num_qubits: int, p: float = 0.1):
    return cirq.kraus(cirq.depolarize(p, num_qubits))


def _split_list(a_list):
    half = len(a_list) // 2
    return a_list[:half], a_list[half:]


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
) -> list[tn.Edge]:
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
