import argparse
import time
from datetime import datetime

import duckdb
import numpy as np

from autobench.params import rows, cols, name, max_k, permutations, runs, epochs, \
    seed
from src.bench import bench


def main(col_decomposition: str, k: int, nr_factors: int):
    np.random.seed(seed)

    results_db_orig = np.zeros((len(rows), len(cols)))
    results_db_kron = np.zeros((len(rows), len(cols)))
    times_db_orig = np.zeros((len(rows), len(cols)))
    times_db_kron = np.zeros((len(rows), len(cols)))
    results_np_orig = np.zeros((len(rows), len(cols)))
    results_np_kron = np.zeros((len(rows), len(cols)))
    times_np_orig = np.zeros((len(rows), len(cols)))
    times_np_kron = np.zeros((len(rows), len(cols)))

    max_rank_suffix = f"_rank_{max_k}"

    config = """
    PRAGMA enable_profiling='json';
    PRAGMA profile_output='out/.temp';
    PRAGMA threads=48;
    """

    cc = col_decomposition == "cc"
    sc = col_decomposition == "sc"

    col_suffix = "_cc" if cc else "_sc" if sc else ""

    start_outer = time.time()
    print(f"{col_decomposition}, rank {k}, {nr_factors} factors", flush=True)
    print(flush=True)
    for r, row in enumerate(rows):
        for c, col in enumerate(cols):
            start = time.time()
            print(f"rows: {row:,}, cols: {col:,}", flush=True)
            permutation_results_db_orig = np.zeros(permutations)
            permutation_results_db_kron = np.zeros(permutations)
            permutation_times_db_orig = np.zeros(permutations)
            permutation_times_db_kron = np.zeros(permutations)
            permutation_results_np_orig = np.zeros(permutations)
            permutation_results_np_kron = np.zeros(permutations)
            permutation_times_np_orig = np.zeros(permutations)
            permutation_times_np_kron = np.zeros(permutations)

            full_name = f"{name}_{row}x{col}"
            mat_a = np.loadtxt(f"data/matrices/{full_name}{col_suffix}_rank_{max_k}_a.csv", delimiter=",")
            mat_b = np.loadtxt(f"data/matrices/{full_name}{col_suffix}_rank_{max_k}_b.csv", delimiter=",")
            mat_c = np.loadtxt(f"data/matrices/{full_name}.csv", delimiter=",")

            database = f"data/databases/{name}_{row}x{col}{col_suffix}{max_rank_suffix}.db"
            con = duckdb.connect(database=database)
            con.execute(config)

            for permutation in range(permutations):
                col_indices = np.random.choice(range(col), nr_factors)
                results_db, times_db, results_np, times_np = bench(name,
                                                                   (row, col),
                                                                   k,
                                                                   col_indices,
                                                                   max_rank=max_k,
                                                                   database=database,
                                                                   sc=sc,
                                                                   cc=cc,
                                                                   runs=runs,
                                                                   epochs=epochs,
                                                                   mat_a=mat_a,
                                                                   mat_b=mat_b,
                                                                   mat_c=mat_c,
                                                                   provided_con=con)
                permutation_results_db_orig[permutation] = results_db[0]
                permutation_results_db_kron[permutation] = results_db[1]
                permutation_times_db_orig[permutation] = times_db[0]
                permutation_times_db_kron[permutation] = times_db[1]
                permutation_results_np_orig[permutation] = results_np[0]
                permutation_results_np_kron[permutation] = results_np[1]
                permutation_times_np_orig[permutation] = times_np[0]
                permutation_times_np_kron[permutation] = times_np[1]

            con.close()

            results_db_orig[r, c] = np.mean(permutation_results_db_orig)
            results_db_kron[r, c] = np.mean(permutation_results_db_kron)
            times_db_orig[r, c] = np.mean(permutation_times_db_orig)
            times_db_kron[r, c] = np.mean(permutation_times_db_kron)
            results_np_orig[r, c] = np.mean(permutation_results_np_orig)
            results_np_kron[r, c] = np.mean(permutation_results_np_kron)
            times_np_orig[r, c] = np.mean(permutation_times_np_orig)
            times_np_kron[r, c] = np.mean(permutation_times_np_kron)

            end = time.time()
            end_datetime = datetime.fromtimestamp(end).strftime('%Y-%m-%d %H:%M:%S')
            elapsed_time = int(end - start)

            print(f"{end_datetime} ({elapsed_time}s)", flush=True)
            print(flush=True)

    results_path = f'data/results/'
    rank_suffix = f"_rank{k}"
    factors_suffix = f"_factors{nr_factors}"
    suffix = f"{factors_suffix}{rank_suffix}{col_suffix}"

    np.savetxt(f'{results_path}results_db_orig{suffix}.csv', results_db_orig, delimiter=',')
    np.savetxt(f'{results_path}results_db_kron{suffix}.csv', results_db_kron, delimiter=',')
    np.savetxt(f'{results_path}times_db_orig{suffix}.csv', times_db_orig, delimiter=',')
    np.savetxt(f'{results_path}times_db_kron{suffix}.csv', times_db_kron, delimiter=',')
    np.savetxt(f'{results_path}results_np_orig{suffix}.csv', results_np_orig, delimiter=',')
    np.savetxt(f'{results_path}results_np_kron{suffix}.csv', results_np_kron, delimiter=',')
    np.savetxt(f'{results_path}times_np_orig{suffix}.csv', times_np_orig, delimiter=',')
    np.savetxt(f'{results_path}times_np_kron{suffix}.csv', times_np_kron, delimiter=',')

    end_outer = time.time()
    print(f"Done! ({int(end_outer - start_outer)}s)", flush=True)


def parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--col_decomposition", "-cd", type=str, default="cc", choices=["cc", "sc", "nc"])
    parser.add_argument("--rank", "-r", "-k", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    parser.add_argument("--factors", "-f", type=int, default=1, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    main(args["col_decomposition"], args["rank"], args["factors"])
