import itertools


def queries(col_indices: list[int], nr_cols: int, rank_k: int, max_rank: int, individual: bool = False):
    col_format_a = "03d" if nr_cols * max_rank > 100 else "02d" if nr_cols * max_rank > 10 else "01d"
    col_format_b = col_format_a if individual else "03d" if max_rank > 100 else "02d" if max_rank > 10 else "01d"
    col_format_c = "03d" if nr_cols > 100 else "02d" if nr_cols > 10 else "01d"

    original_query = f"SELECT SUM({' * '.join([f'column{i:{col_format_c}}' for i in col_indices])}) FROM C;"

    if len(col_indices) == 1:
        # SUM
        col_idx = col_indices[0]
        terms = []
        for r in range(rank_k):
            col_idx_a = r * nr_cols + col_idx
            col_idx_b = r * nr_cols + col_idx if individual else r
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
                col_idx_a = r * nr_cols + col_idx
                col_idx_b = r * nr_cols + col_idx if individual else r
                products_a.append(f"column{col_idx_a:{col_format_a}}")
                products_b.append(f"column{col_idx_b:{col_format_b}}")
            terms.append(f"(SELECT SUM({' * '.join(products_a)}) FROM A) * "
                         f"(SELECT SUM({' * '.join(products_b)}) FROM B)")
        kronecker_query = f"SELECT {' + '.join(terms)};"

    return original_query, kronecker_query
