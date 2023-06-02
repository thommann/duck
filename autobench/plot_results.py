import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows, cols

errors_sum = np.loadtxt('../data/results/sum_errors.csv', delimiter=',')
speedups_sum = np.loadtxt('../data/results/sum_speedups.csv', delimiter=',')
errors_sumproduct = np.loadtxt('../data/results/sumproduct_errors.csv', delimiter=',')
speedups_sumproduct = np.loadtxt('../data/results/sumproduct_speedups.csv', delimiter=',')


def plot(x: list, y: np.ndarray, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(x, y)
    plt.savefig('data/plots/' + filename + '.png')


# plot sum error over rows
plot(rows, errors_sum.mean(axis=1), 'Sum error over rows', 'rows', 'error', 'sum_error_over_rows')

# plot sum error over cols
plot(cols, errors_sum.mean(axis=0), 'Sum error over cols', 'cols', 'error', 'sum_error_over_cols')

# plot sumproduct error over rows
plot(rows, errors_sumproduct.mean(axis=1), 'Sum-product error over rows', 'rows', 'error', 'sumproduct_error_over_rows')

# plot sumproduct error over cols
plot(cols, errors_sumproduct.mean(axis=0), 'Sum-product error over cols', 'cols', 'error', 'sumproduct_error_over_cols')

# plot sum speedup over rows
plot(rows, speedups_sum.mean(axis=1), 'Sum speedup over rows', 'rows', 'speedup', 'sum_speedup_over_rows')

# plot sum speedup over cols
plot(cols, speedups_sum.mean(axis=0), 'Sum speedup over cols', 'cols', 'speedup', 'sum_speedup_over_cols')

# plot sumproduct speedup over rows
plot(rows, speedups_sumproduct.mean(axis=1), 'Sum-product speedup over rows', 'rows', 'speedup',
     'sumproduct_speedup_over_rows')

# plot sumproduct speedup over cols
plot(cols, speedups_sumproduct.mean(axis=0), 'Sum-product speedup over cols', 'cols', 'speedup',
     'sumproduct_speedup_over_cols')

# plot sum error over cols for max rows
plot(cols, errors_sum[-1], 'Sum error over cols for max rows', 'cols', 'error', 'sum_error_over_cols_max_rows')

# plot sumproduct error over cols for max rows
plot(cols, errors_sumproduct[-1], 'Sum-product error over cols for max rows', 'cols', 'error',
     'sumproduct_error_over_cols_max_rows')

# plot sum speedup over cols for max rows
plot(cols, speedups_sum[-1], 'Sum speedup over cols for max rows', 'cols', 'speedup', 'sum_speedup_over_cols_max_rows')

# plot sumproduct speedup over cols for max rows
plot(cols, speedups_sumproduct[-1], 'Sum-product speedup over cols for max rows', 'cols', 'speedup',
     'sumproduct_speedup_over_cols_max_rows')
