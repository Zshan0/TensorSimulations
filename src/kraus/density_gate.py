import tensornetwork as tn
from .pauli import Z

def z(rho):
    """
    Apply z gate to many-qubit system.

    Input:
        - rho: edges of the many-qubit system. [[n edges], [n edges]]

    Returns:
        - Edges for the new density matrix.
    """
    z = Z(False)
    z_t = Z(True)

    # TODO: Very ugly code.
    zs = [tn.replicate_nodes([z])[0] for _ in range(len(rho[0]) - 1)] + [z]
    zs_t = [tn.replicate_nodes([z_t])[0] for _ in range(len(rho[0]) - 1)] + [z_t]

    for (z, z_t, e1, e2) in zip(zs, zs_t, rho[0], rho[1]):
        _ = z[0] ^ e2
        _ = e1 ^ z_t[1]

    return ([z_t[0] for z_t in zs_t], [z[1] for z in zs])
