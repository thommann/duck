import numpy as np

from autobench.params import rows, cols, name, max_k, k
from src.bench_rank_k import bench_rank_k

errors_sum = np.zeros((len(rows), len(cols)))
speedups_sum = np.zeros((len(rows), len(cols)))
errors_sumproduct = np.zeros((len(rows), len(cols)))
speedups_sumproduct = np.zeros((len(rows), len(cols)))

for r, row in enumerate(rows):
    for c, col in enumerate(cols):
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        database = f"data/databases/{name}_{row}x{col}_rank_{max_k}.db"
        error_sum, speedup_sum, error_sumproduct, speedup_sumproduct = bench_rank_k(name, (row, col), k, database)
        errors_sum[r, c] = error_sum
        speedups_sum[r, c] = speedup_sum
        errors_sumproduct[r, c] = error_sumproduct
        speedups_sumproduct[r, c] = speedup_sumproduct

np.savetxt('data/results/sum_errors.csv', errors_sum, delimiter=',')
np.savetxt('data/results/sum_speedups.csv', speedups_sum, delimiter=',')
np.savetxt('data/results/sumproduct_errors.csv', errors_sumproduct, delimiter=',')
np.savetxt('data/results/sumproduct_speedups.csv', speedups_sumproduct, delimiter=',')
