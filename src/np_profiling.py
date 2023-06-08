import itertools
import time

import numpy as np

from src.queries import kronecker_indices


def np_kron_sum(col_idx: int,
                mat_a: np.ndarray,
                mat_b: np.ndarray,
                rank_k: int,
                max_rank: int,
                sc: bool,
                nr_cols_a: int,
                nr_cols_b: int):
    mat_a = np.copy(mat_a, order="F")
    mat_b = np.copy(mat_b, order="F")
    start = time.time_ns()
    result = 0
    for r in range(rank_k):
        col_idx_a, col_idx_b = kronecker_indices(col_idx, nr_cols_a, nr_cols_b, sc, r, max_rank)
        result += np.sum(mat_a[:, col_idx_a]) * np.sum(mat_b[:, col_idx_b])

    end = time.time_ns()
    timing = end - start
    return result, timing


def np_kron_sumproduct(mat_a: np.ndarray,
                       mat_b: np.ndarray,
                       col_indices: list[int],
                       rank_k: int,
                       max_rank: int,
                       sc: bool,
                       nr_cols_a: int,
                       nr_cols_b: int):
    mat_a = np.copy(mat_a, order="F")
    mat_b = np.copy(mat_b, order="F")
    start = time.time_ns()
    combinations = itertools.product(*[itertools.product([idx], range(rank_k)) for idx in col_indices])
    result = 0
    for combination in combinations:
        cols_a = []
        cols_b = []
        for col_idx, r in combination:
            col_idx_a, col_idx_b = kronecker_indices(col_idx, nr_cols_a, nr_cols_b, sc, r, max_rank)
            cols_a.append(mat_a[:, col_idx_a])
            cols_b.append(mat_b[:, col_idx_b])
        result += np.sum(np.prod(cols_a, axis=0)) * np.sum(np.prod(cols_b, axis=0))

    end = time.time_ns()
    timing = end - start
    return result, timing


def np_profiling(mat_a: np.ndarray,
                 mat_b: np.ndarray,
                 mat_c: np.ndarray,
                 nr_cols: int,
                 col_indices: list[int],
                 rank_k: int,
                 max_rank: int,
                 cc: bool,
                 sc: bool,
                 nr_cols_b: int = None,
                 runs: int = 1,
                 epochs: int = 1) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    """
    assert not (sc and cc)
    assert (cc and nr_cols_b is not None) or (not cc and nr_cols_b is None)

    mat_a = np.copy(mat_a, order="F")
    mat_b = np.copy(mat_b, order="F")
    mat_c = np.copy(mat_c, order="F")

    original_timings = []
    kronecker_timings = []

    original_results = []
    kronecker_results = []

    queries = [("original", original_timings, original_results),
               ("kronecker", kronecker_timings, kronecker_results)]

    nr_cols_b = nr_cols if sc else nr_cols_b if cc else 1
    nr_cols_a = nr_cols if sc else nr_cols // nr_cols_b
    for epoch in range(epochs):
        for query, timings, results in queries:
            for run in range(runs):
                if query == "original":
                    # Original
                    start = time.time_ns()
                    result = np.sum(np.prod(mat_c[:, col_indices], axis=1))
                    end = time.time_ns()
                    timing = end - start
                    timings.append(timing)
                    results.append(result)

                else:
                    if len(col_indices) == 1:
                        # SUM
                        col_idx = col_indices[0]
                        result, timing = np_kron_sum(col_idx, mat_a, mat_b, rank_k, max_rank, sc, nr_cols_a, nr_cols_b)
                        timings.append(timing)
                        results.append(result)

                    else:
                        # SUM product
                        result, timing = np_kron_sumproduct(mat_a, mat_b, col_indices, rank_k, max_rank, sc, nr_cols_a,
                                                            nr_cols_b)
                        timings.append(timing)
                        results.append(result)

    average_original_time = float(np.mean(original_timings))
    average_kronecker_time = float(np.mean(kronecker_timings))
    assert len(set(original_results)) == 1
    assert len(set(kronecker_results)) == 1

    return (original_results[0], kronecker_results[0]), (average_original_time, average_kronecker_time)
