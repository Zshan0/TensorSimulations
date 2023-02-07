import numpy as np
import tensornetwork as tn

i = np.array([[1, 0], [0, 1]])
z = np.array([[1, 0], [0, -1]])
x = np.array([[0, 1], [1, 0]])
y = np.array([[0j, -1j], [1j, 0j]], dtype=complex)
inds = [i, x, y, z]


class Pauli(tn.Node):
    """
    Index Mapping: I X Y Z
    """

    def __init__(self, ind: int = 0, is_conj=False, label=""):
        if not is_conj:
            super().__init__(inds[ind], name="Pauli_" + label)
        else:
            super().__init__(inds[ind].conj().T, name="Pauli_T_" + label)


class Z(Pauli):
    def __init__(self, is_conj=False, label=""):
        super().__init__(3, is_conj=is_conj, label="Z_" + label)
