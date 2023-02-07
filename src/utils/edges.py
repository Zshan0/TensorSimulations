import tensornetwork as tn

def _split_and_alternate(row:tn.Edge, col:tn.Edge, qubit_count:int) -> list[tn.Edge]:
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
