import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows, cols, cc, sc, k, max_k, nr_factors

assert k <= max_k

results_path = f'data/results/'
factors_suffix = f"_factors{nr_factors}"
rank_suffix = f"_rank{k}"
col_suffix = "_cc" if cc else "_sc" if sc else ""
suffix = f'{factors_suffix}{rank_suffix}{col_suffix}'

results_orig = np.loadtxt(f'{results_path}results_orig{suffix}.csv', delimiter=',')
results_kron = np.loadtxt(f'{results_path}results_kron{suffix}.csv', delimiter=',')
times_orig = np.loadtxt(f'{results_path}times_orig{suffix}.csv', delimiter=',')
times_kron = np.loadtxt(f'{results_path}times_kron{suffix}.csv', delimiter=',')


def plot(y: np.ndarray, title: str, ylabel: str, filename: str, ylog: bool = False) -> None:
    plt.figure()
    plt.title(title)
    plt.xlabel('rows')
    plt.ylabel(ylabel)
    plt.xscale('log')
    if ylog:
        plt.yscale('log')
    for i, col in enumerate(cols):
        plt.plot(rows, y[:, i], label=f"{col} cols")
    plt.legend()
    plt.savefig('data/plots/' + filename + '.png')


operation_name = "SUM" if nr_factors == 1 else "SUM-product"
col_method_param = "single, " if sc else "compressed, " if cc else ""
factors_param = f"{nr_factors} factors, " if nr_factors > 1 else ""

# plot relative error
relative_errors_sum = np.abs(results_orig - results_kron) / np.abs(results_orig)
plot(relative_errors_sum,
     f'{operation_name} Rel. Error ({factors_param}{col_method_param}rank {k}, )',
     'Rel. Error',
     f'rel_error{suffix}',
     ylog=True)
# plot absolute error
absolute_errors_sum = np.abs(results_orig - results_kron)
plot(absolute_errors_sum,
     f'{operation_name} Abs. Error ({factors_param}{col_method_param}{col_method_param}rank {k})',
     'Abs. Error',
     f'abs_error{suffix}')
# plot speedup
speedups_sum = times_orig / times_kron
plot(speedups_sum,
     f'{operation_name} Speedup ({factors_param}{col_method_param}rank {k})',
     'Speedup',
     f'speedup{suffix}')

print("All Done!")
