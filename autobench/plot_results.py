import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows, cols, compress_cols, single_column, k, max_k

assert k <= max_k

col_suffix = "_cc" if compress_cols else "_sc" if single_column else ""
rank_suffix = f"_rank_{k}" if k > 1 else ""

results_sum_orig = np.loadtxt(f'data/results/results_sum_orig{col_suffix}{rank_suffix}.csv', delimiter=',')
results_sum_kron = np.loadtxt(f'data/results/results_sum_kron{col_suffix}{rank_suffix}.csv', delimiter=',')
times_sum_orig = np.loadtxt(f'data/results/times_sum_orig{col_suffix}{rank_suffix}.csv', delimiter=',')
times_sum_kron = np.loadtxt(f'data/results/times_sum_kron{col_suffix}{rank_suffix}.csv', delimiter=',')
results_sumproduct_orig = np.loadtxt(f'data/results/results_sumproduct_orig{col_suffix}{rank_suffix}.csv',
                                     delimiter=',')
results_sumproduct_kron = np.loadtxt(f'data/results/results_sumproduct_kron{col_suffix}{rank_suffix}.csv',
                                     delimiter=',')
times_sumproduct_orig = np.loadtxt(f'data/results/times_sumproduct_orig{col_suffix}{rank_suffix}.csv', delimiter=',')
times_sumproduct_kron = np.loadtxt(f'data/results/times_sumproduct_kron{col_suffix}{rank_suffix}.csv', delimiter=',')


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
# plot relative error sum
relative_errors_sum = np.abs(results_sum_orig - results_sum_kron) / np.abs(results_sum_orig)
plot(relative_errors_sum, f'Sum Rel. Error (rank {k}, {col_method_param})', 'Rel. Error',
     f'sum_error{col_suffix}{rank_suffix}')
# plot speedup sum
speedups_sum = times_sum_orig / times_sum_kron
plot(speedups_sum, f'Sum Speedup (rank {k}, {col_method_param})', 'Speedup', f'sum_speedup{col_suffix}{rank_suffix}')
# plot relative error sumproduct
relative_errors_sumproduct = np.abs(results_sumproduct_orig - results_sumproduct_kron) / np.abs(
    results_sumproduct_orig)
plot(relative_errors_sumproduct, f'SumProduct Rel. Error (rank {k}, {col_method_param})', 'Rel. Error',
     f'sumproduct_error{col_suffix}{rank_suffix}')
# plot speedup sumproduct
speedups_sumproduct = times_sumproduct_orig / times_sumproduct_kron
plot(speedups_sumproduct, f'SumProduct Speedup (rank {k}, {col_method_param})', 'Speedup',
     f'sumproduct_speedup{col_suffix}{rank_suffix}')
