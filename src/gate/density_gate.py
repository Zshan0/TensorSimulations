from .pauli import Z


def z(rho):
    """
    Apply z gate to many-qubit system.

    Input:
        - rho: edges of the many-qubit system. [[n edges], [n edges]]

    Returns:
        - Edges for the new density matrix.
    """
    zs = [Z(False) for _ in range(len(rho[0]))]
    zs_t = [Z(True) for _ in range(len(rho[0]))]

    for (z, z_t, e1, e2) in zip(zs, zs_t, rho[0], rho[1]):
        _ = z[0] ^ e2
        _ = e1 ^ z_t[1]

    return [z_t[0] for z_t in zs_t] + [z[1] for z in zs]
