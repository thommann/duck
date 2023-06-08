import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows, cols, cc, sc, k, max_k, nr_factors

assert k <= max_k

results_path = f'data/results/'
factors_suffix = f"_factors{nr_factors}"
rank_suffix = f"_rank{k}"
col_suffix = "_cc" if cc else "_sc" if sc else ""
suffix = f'{factors_suffix}{rank_suffix}{col_suffix}'

results_db_orig = np.loadtxt(f'{results_path}results_db_orig{suffix}.csv', delimiter=',')
results_db_kron = np.loadtxt(f'{results_path}results_db_kron{suffix}.csv', delimiter=',')
times_db_orig = np.loadtxt(f'{results_path}times_db_orig{suffix}.csv', delimiter=',')
times_db_kron = np.loadtxt(f'{results_path}times_db_kron{suffix}.csv', delimiter=',')

results_np_orig = np.loadtxt(f'{results_path}results_np_orig{suffix}.csv', delimiter=',')
results_np_kron = np.loadtxt(f'{results_path}results_np_kron{suffix}.csv', delimiter=',')
times_np_orig = np.loadtxt(f'{results_path}times_np_orig{suffix}.csv', delimiter=',')
times_np_kron = np.loadtxt(f'{results_path}times_np_kron{suffix}.csv', delimiter=',')


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

# plot relative error db
relative_errors_db = np.abs(results_db_orig - results_db_kron) / np.abs(results_db_orig)
plot(relative_errors_db,
     f'DB {operation_name} Rel. Error ({factors_param}{col_method_param}rank {k}, )',
     'Rel. Error',
     f'rel_error_db{suffix}',
     ylog=True)
# plot absolute error db
absolute_errors_db = np.abs(results_db_orig - results_db_kron)
plot(absolute_errors_db,
     f'DB {operation_name} Abs. Error ({factors_param}{col_method_param}{col_method_param}rank {k})',
     'Abs. Error',
     f'abs_error_db{suffix}')
# plot speedup db
speedups_db = times_db_orig / times_db_kron
plot(speedups_db,
     f'DB {operation_name} Speedup ({factors_param}{col_method_param}rank {k})',
     'Speedup',
     f'speedup_db{suffix}')

# plot relative error np
relative_errors_np = np.abs(results_np_orig - results_np_kron) / np.abs(results_np_orig)
plot(relative_errors_np,
     f'NP {operation_name} Rel. Error ({factors_param}{col_method_param}rank {k})',
     'Rel. Error',
     f'rel_error_np{suffix}',
     ylog=True)
# plot absolute error np
absolute_errors_np = np.abs(results_np_orig - results_np_kron)
plot(absolute_errors_np,
     f'NP {operation_name} Abs. Error ({factors_param}{col_method_param}rank {k})',
     'Abs. Error',
     f'abs_error_np{suffix}')
# plot speedup np
speedups_np = times_np_orig / times_np_kron
plot(speedups_np,
     f'NP {operation_name} Speedup ({factors_param}{col_method_param}rank {k})',
     'Speedup',
     f'speedup_np{suffix}')

print("All Done!")
