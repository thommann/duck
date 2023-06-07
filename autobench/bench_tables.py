import time

import numpy as np

from autobench.params import rows, cols, name, max_k, k, compress_cols, single_column
from src.bench import bench
from src.bench_sc import bench_sc

assert not (compress_cols and single_column)
assert k <= max_k

results_sum_orig = np.zeros((len(rows), len(cols)))
results_sum_kron = np.zeros((len(rows), len(cols)))
times_sum_orig = np.zeros((len(rows), len(cols)))
times_sum_kron = np.zeros((len(rows), len(cols)))
results_sumproduct_orig = np.zeros((len(rows), len(cols)))
results_sumproduct_kron = np.zeros((len(rows), len(cols)))
times_sumproduct_orig = np.zeros((len(rows), len(cols)))
times_sumproduct_kron = np.zeros((len(rows), len(cols)))

col_suffix = "_cc" if compress_cols else "_sc" if single_column else ""
max_rank_suffix = f"_rank_{max_k}"

for r, row in enumerate(rows):
    for c, col in enumerate(cols):
        start = time.time()
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        database = f"data/databases/{name}_{row}x{col}{col_suffix}{max_rank_suffix}.db"
        if single_column:
            sum_results, sum_times, sumproduct_results, sumproduct_times = bench_sc(name, (row, col), k,
                                                                                    max_rank=max_k,
                                                                                    database=database)
        else:
            sum_results, sum_times, sumproduct_results, sumproduct_times = bench(name, (row, col), k,
                                                                                 max_rank=max_k,
                                                                                 database=database,
                                                                                 cc=compress_cols)
        results_sum_orig[r, c] = sum_results[0]
        results_sum_kron[r, c] = sum_results[1]
        times_sum_orig[r, c] = sum_times[0]
        times_sum_kron[r, c] = sum_times[1]
        results_sumproduct_orig[r, c] = sumproduct_results[0]
        results_sumproduct_kron[r, c] = sumproduct_results[1]
        times_sumproduct_orig[r, c] = sumproduct_times[0]
        times_sumproduct_kron[r, c] = sumproduct_times[1]
        end = time.time()

        print(f"Done! ({int(end - start)}s)", flush=True)
        print()

rank_suffix = f"_rank_{k}"
np.savetxt(f'data/results/results_sum_orig{col_suffix}{rank_suffix}.csv', results_sum_orig, delimiter=',')
np.savetxt(f'data/results/results_sum_kron{col_suffix}{rank_suffix}.csv', results_sum_kron, delimiter=',')
np.savetxt(f'data/results/results_sumproduct_orig{col_suffix}{rank_suffix}.csv', results_sumproduct_orig,
           delimiter=',')
np.savetxt(f'data/results/results_sumproduct_kron{col_suffix}{rank_suffix}.csv', results_sumproduct_kron,
           delimiter=',')
np.savetxt(f'data/results/times_sum_orig{col_suffix}{rank_suffix}.csv', times_sum_orig, delimiter=',')
np.savetxt(f'data/results/times_sum_kron{col_suffix}{rank_suffix}.csv', times_sum_kron, delimiter=',')
np.savetxt(f'data/results/times_sumproduct_orig{col_suffix}{rank_suffix}.csv', times_sumproduct_orig, delimiter=',')
np.savetxt(f'data/results/times_sumproduct_kron{col_suffix}{rank_suffix}.csv', times_sumproduct_kron, delimiter=',')

print("All done!")
