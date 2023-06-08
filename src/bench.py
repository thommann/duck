import numpy as np

from src.db_profiling import query_results, query_profiling
from src.np_profiling import np_profiling
from src.queries import queries


def bench(
        name: str,
        dimensions: tuple[int, int],
        rank_k: int,
        col_indices: list[int],
        max_rank: int = 10,
        database: str | None = None,
        sc: bool = False,
        cc: bool = False,
        runs: int = 1,
        epochs: int = 1,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    """
    Bench a single matrix.
    :param epochs: how many times to run the benchmark with both methods
    :param runs: how many times to run the benchmark per method per epoch
    :param name: Name of the matrix to benchmark
    :param dimensions: Dimensions of the matrix
    :param rank_k: Rank of the matrix
    :param col_indices: Indices of the columns to benchmark
    :param max_rank: Maximum rank to benchmark
    :param database: Database to use
    :param sc: Whether to use single columns
    :param cc: Whether to use column compression
    :return: Results of the database (original, kronecker),
                times of the database (original, kronecker),
                results of numpy (original, kronecker),
                times of numpy (original, kronecker),
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

    db_results = query_results(original, kronecker, database)
    db_times = query_profiling(original, kronecker, database, runs=runs, epochs=epochs)

    mat_a = np.loadtxt(f"data/matrices/{full_name}{suffix}_rank_{max_rank}_a.csv", delimiter=",")
    mat_b = np.loadtxt(f"data/matrices/{full_name}{suffix}_rank_{max_rank}_b.csv", delimiter=",")
    mat_c = np.loadtxt(f"data/matrices/{full_name}.csv", delimiter=",")
    np_results, np_times = np_profiling(mat_a, mat_b, mat_c, nr_cols, col_indices, rank_k, max_rank, cc, sc, nr_cols_b,
                                        runs=runs, epochs=epochs)

    return db_results, db_times, np_results, np_times
