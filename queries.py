import itertools


def queries(col_indices: list[int], nr_cols: int, rank_k: int, max_rank: int):
    col_format_a = "03d" if nr_cols * max_rank > 100 else "02d" if nr_cols * max_rank > 10 else "01d"
    col_format_b = "03d" if max_rank > 100 else "02d" if max_rank > 10 else "01d"
    col_format_c = "03d" if nr_cols > 100 else "02d" if nr_cols > 10 else "01d"

    original_query = f"SELECT SUM({' * '.join([f'column{i:{col_format_c}}' for i in col_indices])}) FROM C;"

    if len(col_indices) == 1:
        # SUM
        col_idx = col_indices[0]
        kronecker_query = f"SELECT {' + '.join([f'(SELECT SUM(column{r * nr_cols + col_idx:{col_format_a}}) FROM A) * (SELECT SUM(column{r:{col_format_b}}) FROM B)' for r in range(rank_k)])};"

    else:
        # SUM product
        terms = itertools.product(*[itertools.product([idx], range(rank_k)) for idx in col_indices])
        kronecker_query = f"""SELECT {' + '.join([f'(SELECT SUM({" * ".join([f"column{r * nr_cols + col_idx:{col_format_a}}" for col_idx, r in term])}) FROM A) * (SELECT SUM({" * ".join([f"column{r:{col_format_b}}" for _, r in term])}) FROM B)' for term in terms])};"""

    return original_query, kronecker_query
