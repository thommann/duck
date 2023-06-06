import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows, cols, compress_cols, single_column, k, max_k

assert k <= max_k

col_suffix = "_cc" if compress_cols else "_sc" if single_column else ""
rank_suffix = f"_rank_{k}" if k > 1 else ""

errors_sum = np.loadtxt(f'data/results/sum_errors{col_suffix}{rank_suffix}.csv', delimiter=',')
speedups_sum = np.loadtxt(f'data/results/sum_speedups{col_suffix}{rank_suffix}.csv', delimiter=',')
errors_sumproduct = np.loadtxt(f'data/results/sumproduct_errors{col_suffix}{rank_suffix}.csv', delimiter=',')
speedups_sumproduct = np.loadtxt(f'data/results/sumproduct_speedups{col_suffix}{rank_suffix}.csv', delimiter=',')


def plot(y: np.ndarray, title: str, ylabel: str, filename: str) -> None:
    plt.figure()
    plt.title(title)
    plt.xlabel('rows')
    plt.ylabel(ylabel)
    plt.xscale('log')
    for i, col in enumerate(cols):
        plt.plot(rows, y[:, i], label=f"{col} cols")
    plt.legend()
    plt.savefig('data/plots/' + filename + '.png')


col_method_param = "single" if single_column else "compressed" if compress_cols else "uncompressed"
# plot sum error
plot(errors_sum, f'Sum Error (rank {k}, {col_method_param})', 'Error', f'sum_error{col_suffix}{rank_suffix}')
# plot sum speedup
plot(speedups_sum, f'Sum Speedup (rank {k}, {col_method_param})', 'Speedup', f'sum_speedup{col_suffix}{rank_suffix}')
