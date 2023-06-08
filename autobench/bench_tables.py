import time

import numpy as np

from autobench.params import rows, cols, name, max_k, k, cc, sc, permutations, nr_factors, runs, epochs
from src.bench import bench

assert not (cc and sc)
assert k <= max_k

results_db_orig = np.zeros((len(rows), len(cols)))
results_db_kron = np.zeros((len(rows), len(cols)))
times_db_orig = np.zeros((len(rows), len(cols)))
times_db_kron = np.zeros((len(rows), len(cols)))
results_np_orig = np.zeros((len(rows), len(cols)))
results_np_kron = np.zeros((len(rows), len(cols)))
times_np_orig = np.zeros((len(rows), len(cols)))
times_np_kron = np.zeros((len(rows), len(cols)))

col_suffix = "_cc" if cc else "_sc" if sc else ""
max_rank_suffix = f"_rank_{max_k}"

for r, row in enumerate(rows):
    start = time.time()
    for c, col in enumerate(cols):
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
            print(f"PERMUTATION {permutation + 1}/{permutations}", flush=True, end="\r")
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
    print(f"Done! ({int(end - start)}s)", flush=True)
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

print("All done!")
