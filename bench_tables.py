import numpy as np

from bench import bench

rows = [2 ** x for x in range(10, 26 + 1)]
cols = [2 ** x for x in range(1, 4 + 1)]

name = 'webb'

errors_sum = np.zeros((len(rows), len(cols)))
speedups_sum = np.zeros((len(rows), len(cols)))
errors_sumproduct = np.zeros((len(rows), len(cols)))
speedups_sumproduct = np.zeros((len(rows), len(cols)))

for r in range(len(rows)):
    row = rows[r]
    for c in range(len(cols)):
        col = cols[c]
        print(f"row: {row}, col: {col}")
        error_sum, speedup_sum, error_sumproduct, speedup_sumproduct = bench(name, (row, col))
        errors_sum[r, c] = error_sum
        speedups_sum[r, c] = speedup_sum
        errors_sumproduct[r, c] = error_sumproduct
        speedups_sumproduct[r, c] = speedup_sumproduct
        print(flush=True)

np.savetxt('data/results/sum_errors.csv', errors_sum, delimiter=',')
np.savetxt('data/results/sum_speedups.csv', speedups_sum, delimiter=',')
np.savetxt('data/results/sumproduct_errors.csv', errors_sumproduct, delimiter=',')
np.savetxt('data/results/sumproduct_speedups.csv', speedups_sumproduct, delimiter=',')
