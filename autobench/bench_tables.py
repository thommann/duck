import time
from datetime import datetime

import numpy as np

from autobench.params import rows, cols, name, max_k, ks, permutations, factors, runs, epochs, \
    col_decompositions
from src.bench import bench

results_db_orig = np.zeros((len(rows), len(cols)))
results_db_kron = np.zeros((len(rows), len(cols)))
times_db_orig = np.zeros((len(rows), len(cols)))
times_db_kron = np.zeros((len(rows), len(cols)))
results_np_orig = np.zeros((len(rows), len(cols)))
results_np_kron = np.zeros((len(rows), len(cols)))
times_np_orig = np.zeros((len(rows), len(cols)))
times_np_kron = np.zeros((len(rows), len(cols)))

max_rank_suffix = f"_rank_{max_k}"

for col_decomposition in col_decompositions:
    cc = col_decomposition == "cc"
    sc = col_decomposition == "sc"

    col_suffix = "_cc" if cc else "_sc" if sc else ""

    for k in ks:
        for nr_factors in factors:
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
                    for permutation in range(permutations):
                        col_indices = np.random.choice(range(col), nr_factors)
                        database = f"data/databases/{name}_{row}x{col}{col_suffix}{max_rank_suffix}.db"
                        results_db, times_db, results_np, times_np = bench(name,
                                                                           (row, col),
                                                                           k,
                                                                           col_indices,
                                                                           max_rank=max_k,
                                                                           database=database,
                                                                           sc=sc,
                                                                           cc=cc,
                                                                           runs=runs,
                                                                           epochs=epochs)
                        permutation_results_db_orig[permutation] = results_db[0]
                        permutation_results_db_kron[permutation] = results_db[1]
                        permutation_times_db_orig[permutation] = times_db[0]
                        permutation_times_db_kron[permutation] = times_db[1]
                        permutation_results_np_orig[permutation] = results_np[0]
                        permutation_results_np_kron[permutation] = results_np[1]
                        permutation_times_np_orig[permutation] = times_np[0]
                        permutation_times_np_kron[permutation] = times_np[1]

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
            print(flush=True)

print("All done!")
