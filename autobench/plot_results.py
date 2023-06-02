import matplotlib.pyplot as plt
import numpy as np

from autobench.params import rows_base_2, cols

rows = rows_base_2

errors_sum = np.loadtxt('data/results/sum_errors.csv', delimiter=',')
speedups_sum = np.loadtxt('data/results/sum_speedups.csv', delimiter=',')
errors_sumproduct = np.loadtxt('data/results/sumproduct_errors.csv', delimiter=',')
speedups_sumproduct = np.loadtxt('data/results/sumproduct_speedups.csv', delimiter=',')


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


# plot sum error
plot(errors_sum, 'Sum Error', 'Error', 'sum_error')
# plot sum speedup
plot(speedups_sum, 'Sum Speedup', 'Speedup', 'sum_speedup')
