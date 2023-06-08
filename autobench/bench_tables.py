import time

import numpy as np

from autobench.params import rows, cols, name, max_k, k, cc, sc, runs, nr_factors
from src.bench import bench

assert not (cc and sc)
assert k <= max_k

results_orig = np.zeros((len(rows), len(cols)))
results_kron = np.zeros((len(rows), len(cols)))
times_orig = np.zeros((len(rows), len(cols)))
times_kron = np.zeros((len(rows), len(cols)))

col_suffix = "_cc" if cc else "_sc" if sc else ""
max_rank_suffix = f"_rank_{max_k}"

for r, row in enumerate(rows):
    for c, col in enumerate(cols):
        for run in runs:
            col_indices = np.random.choice(range(col), nr_factors)
            start = time.time()
            print(f"rows: {row:,}, cols: {col:,}", flush=True)
            database = f"data/databases/{name}_{row}x{col}{col_suffix}{max_rank_suffix}.db"
            results, times = bench(name,
                                   (row, col),
                                   k,
                                   col_indices,
                                   max_rank=max_k,
                                   database=database,
                                   sc=sc,
                                   cc=cc)
            results_orig[r, c] = results[0]
            results_kron[r, c] = results[1]
            times_orig[r, c] = times[0]
            times_kron[r, c] = times[1]
            end = time.time()

            print(f"Done! ({int(end - start)}s)", flush=True)
            print()

results_path = f'data/results/'
rank_suffix = f"_rank{k}"
factors_suffix = f"_factors{nr_factors}"
suffix = f"{factors_suffix}{rank_suffix}{col_suffix}"

np.savetxt(f'{results_path}results_orig{suffix}.csv', results_orig, delimiter=',')
np.savetxt(f'{results_path}results_kron{suffix}.csv', results_kron, delimiter=',')
np.savetxt(f'{results_path}times_orig{suffix}.csv', times_orig, delimiter=',')
np.savetxt(f'{results_path}times_kron{suffix}.csv', times_kron, delimiter=',')

print("All done!")
