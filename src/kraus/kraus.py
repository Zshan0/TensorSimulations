import tensornetwork as tn

import numpy as np
import tensornetwork as tn
from ..utils import _split_list, _split_and_alternate 


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
