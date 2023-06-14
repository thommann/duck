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
            nr_cols_b: int = None,
            table_a: str = "A",
            table_b: str = "B",
            table_c: str = "C") -> tuple[str, str]:
    assert not (sc and cc)
    assert (cc and nr_cols_b is not None) or (not cc and nr_cols_b is None)

    nr_cols_b = nr_cols if sc else nr_cols_b if cc else 1
    nr_cols_a = nr_cols if sc else nr_cols // nr_cols_b

    col_format_a = f"{len(str(int(nr_cols_a * max_rank - 1))):02d}d"
    col_format_b = f"{len(str(int(nr_cols_b * max_rank - 1))):02d}d"
    col_format_c = f"{len(str(int(nr_cols - 1))):02d}d"

    original_query = f"SELECT SUM({' * '.join([f'column{i:{col_format_c}}' for i in col_indices])}) FROM {table_c};"

    combinations = itertools.product(*[itertools.product([idx], range(rank_k)) for idx in col_indices])
    terms = []
    for combination in combinations:
        products_a = []
        products_b = []
        for col_idx, r in combination:
            col_idx_a, col_idx_b = kronecker_indices(col_idx, nr_cols_a, nr_cols_b, sc, r, max_rank)
            products_a.append(f"column{col_idx_a:{col_format_a}}")
            products_b.append(f"column{col_idx_b:{col_format_b}}")
        terms.append(f"(SELECT SUM({' * '.join(products_a)}) FROM {table_a}) * "
                     f"(SELECT SUM({' * '.join(products_b)}) FROM {table_b})")
    kronecker_query = f"SELECT {' + '.join(terms)};"

    return original_query, kronecker_query
