import time

import numpy as np

from autobench.params import rows, cols, name, max_k, k, compress_cols, single_column
from src.bench import bench

assert not (compress_cols and single_column)
assert k <= max_k

errors_sum = np.zeros((len(rows), len(cols)))
speedups_sum = np.zeros((len(rows), len(cols)))
errors_sumproduct = np.zeros((len(rows), len(cols)))
speedups_sumproduct = np.zeros((len(rows), len(cols)))

col_suffix = "_cc" if compress_cols else "_sc" if single_column else ""
max_rank_suffix = f"_rank_{max_k}" if max_k > 1 else ""

for r, row in enumerate(rows):
    for c, col in enumerate(cols):
        start = time.time()
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        database = f"data/databases/{name}_{row}x{col}{col_suffix}{max_rank_suffix}.db"
        error_sum, speedup_sum, error_sumproduct, speedup_sumproduct = bench(name, (row, col), k,
                                                                             max_rank=max_k, database=database)
        errors_sum[r, c] = error_sum
        speedups_sum[r, c] = speedup_sum
        errors_sumproduct[r, c] = error_sumproduct
        speedups_sumproduct[r, c] = speedup_sumproduct
        end = time.time()
        print(f"Done! ({int(end - start)}s)", flush=True)

rank_suffix = f"_rank_{k}" if k > 1 else ""
np.savetxt(f'data/results/sum_errors{col_suffix}{rank_suffix}.csv', errors_sum, delimiter=',')
np.savetxt(f'data/results/sum_speedups{col_suffix}{rank_suffix}.csv', speedups_sum, delimiter=',')
np.savetxt(f'data/results/sumproduct_errors{col_suffix}{rank_suffix}.csv', errors_sumproduct, delimiter=',')
np.savetxt(f'data/results/sumproduct_speedups{col_suffix}{rank_suffix}.csv', speedups_sumproduct, delimiter=',')
