import numpy as np

from src.profiling import query_results, query_profiling
from src.queries import queries


def bench(name: str,
          dimensions: tuple[int, int],
          rank_k: int,
          col_indices: list[int],
          max_rank: int = 10,
          database: str | None = None,
          sc: bool = False,
          cc: bool = False) -> tuple[tuple, tuple]:
    """
    Bench a single matrix.
    :param name: Name of the matrix to benchmark
    :param dimensions: Dimensions of the matrix
    :param rank_k: Rank of the matrix
    :param col_indices: Indices of the columns to benchmark
    :param max_rank: Maximum rank to benchmark
    :param database: Database to use
    :param sc: Whether to use single columns
    :param cc: Whether to use column compression
    :return: Tuple of results and times
    """
    assert not (cc and sc)

    nr_rows, nr_cols = dimensions
    full_name = f"{name}_{nr_rows}x{nr_cols}"

    nr_cols_b = None
    if cc:
        # If the columns are compressed, we need to know how many columns there are in each kronecker matrix
        nr_cols_b = np.loadtxt(f"data/bcols/{full_name}.csv", delimiter=",", dtype=int)

    suffix = "_cc" if cc else "_sc" if sc else ""
    if database is None:
        database = f"data/databases/{full_name}{suffix}_rank_{max_rank}.db"

    original, kronecker = queries(col_indices, nr_cols, rank_k, max_rank, cc, sc, nr_cols_b)

    sum_results = query_results(original, kronecker, database)
    sum_times = query_profiling(original, kronecker, database)

    return sum_results, sum_times
