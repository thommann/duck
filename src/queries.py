import itertools


def kronecker_indices(col_idx: int, nr_cols_a: int, nr_cols_b: int, sc: bool, r: int, max_rank: int) -> tuple[int, int]:
    if sc:
        col_idx_a = col_idx_b = col_idx * max_rank + r
    else:
        col_idx_a = col_idx // nr_cols_b
        col_idx_b = col_idx % nr_cols_b
        col_idx_a = col_idx_a + r * nr_cols_a
        col_idx_b = col_idx_b + r * nr_cols_b
    return col_idx_a, col_idx_b


def queries(col_indices: list[int],
            nr_cols: int,
            rank_k: int,
            max_rank: int,
            sc: bool = False,
            cc: bool = False,
            nr_cols_b: int = None):
    """
    Generates the original and Kronecker queries for the given column indices, number of columns and rank.
    :param col_indices: list of column indices
    :param nr_cols: number of columns in the input matrices
    :param rank_k: rank of the input matrices
    :param max_rank: maximum rank of the input matrices
    :param sc: whether to use single-column Kronecker queries
    :param cc: whether to use compressed-column Kronecker queries
    :param nr_cols_b: number of columns in the second kronecker matrix (only used if cc is True)
    :return: original and Kronecker queries
    """
    assert not (sc and cc)
    assert (cc and nr_cols_b is not None) or (not cc and nr_cols_b is None)

    nr_cols_b = nr_cols if sc else nr_cols_b if cc else 1
    nr_cols_a = nr_cols if sc else nr_cols // nr_cols_b

    col_format_a = "03d" if nr_cols_a * max_rank > 100 else "02d" if nr_cols_a * max_rank > 10 else "01d"
    col_format_b = "03d" if nr_cols_b * max_rank > 100 else "02d" if nr_cols_b * max_rank > 10 else "01d"
    col_format_c = "03d" if nr_cols > 100 else "02d" if nr_cols > 10 else "01d"

    original_query = f"SELECT SUM({' * '.join([f'column{i:{col_format_c}}' for i in col_indices])}) FROM C;"

    if len(col_indices) == 1:
        # SUM
        col_idx = col_indices[0]
        terms = []
        for r in range(rank_k):
            col_idx_a, col_idx_b = kronecker_indices(col_idx, nr_cols_a, nr_cols_b, sc, r, max_rank)
            terms.append(f"(SELECT SUM(column{col_idx_a:{col_format_a}}) FROM A) * "
                         f"(SELECT SUM(column{col_idx_b:{col_format_b}}) FROM B)")
        kronecker_query = f"SELECT {' + '.join(terms)};"

    else:
        # SUM product
        combinations = itertools.product(*[itertools.product([idx], range(rank_k)) for idx in col_indices])
        terms = []
        for combination in combinations:
            products_a = []
            products_b = []
            for col_idx, r in combination:
                col_idx_a, col_idx_b = kronecker_indices(col_idx, nr_cols_a, nr_cols_b, sc, r, max_rank)
                products_a.append(f"column{col_idx_a:{col_format_a}}")
                products_b.append(f"column{col_idx_b:{col_format_b}}")
            terms.append(f"(SELECT SUM({' * '.join(products_a)}) FROM A) * "
                         f"(SELECT SUM({' * '.join(products_b)}) FROM B)")
        kronecker_query = f"SELECT {' + '.join(terms)};"

    return original_query, kronecker_query
