import numpy as np
import tensornetwork as tn

tenz = np.array([[1, 0], [0, -1]])
tenz_z = tenz.conj().T

class Z(tn.Node):
    def __init__(self, is_conj=False, label=""):
        if not is_conj:
            super().__init__(tenz, name="PauliZ" + label)
        else:
            super().__init__(tenz.conj().T, name="PauliZ_T" + label)


