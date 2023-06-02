import numpy as np

from autobench.params import rows, cols, name
from src.bench import bench

errors_sum = np.zeros((len(rows), len(cols)))
speedups_sum = np.zeros((len(rows), len(cols)))
errors_sumproduct = np.zeros((len(rows), len(cols)))
speedups_sumproduct = np.zeros((len(rows), len(cols)))

for r in range(len(rows)):
    row = rows[r]
    for c in range(len(cols)):
        col = cols[c]
        print(f"rows: {row:,}, cols: {col:,}", flush=True)
        error_sum, speedup_sum, error_sumproduct, speedup_sumproduct = bench(name, (row, col))
        errors_sum[r, c] = error_sum
        speedups_sum[r, c] = speedup_sum
        errors_sumproduct[r, c] = error_sumproduct
        speedups_sumproduct[r, c] = speedup_sumproduct

np.savetxt('data/results/sum_errors.csv', errors_sum, delimiter=',')
np.savetxt('data/results/sum_speedups.csv', speedups_sum, delimiter=',')
np.savetxt('data/results/sumproduct_errors.csv', errors_sumproduct, delimiter=',')
np.savetxt('data/results/sumproduct_speedups.csv', speedups_sumproduct, delimiter=',')
