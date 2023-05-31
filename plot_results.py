import matplotlib.pyplot as plt
import numpy as np

rows = [2 ** x for x in range(1, 24 + 1)]
cols = [2 ** x for x in range(1, 6 + 1)]

errors_sum = np.loadtxt('data/results/sum_errors.csv', delimiter=',')
speedups_sum = np.loadtxt('data/results/sum_speedups.csv', delimiter=',')
errors_sumproduct = np.loadtxt('data/results/sumproduct_errors.csv', delimiter=',')
speedups_sumproduct = np.loadtxt('data/results/sumproduct_speedups.csv', delimiter=',')


def plot(x: list, y: np.ndarray, title: str, xlabel: str, ylabel: str):
    plt.figure()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xscale('log', basex=2)
    plt.yscale('log', basey=10)
    plt.plot(x, y)


# plot sum error over rows
plot(rows, errors_sum.mean(axis=1), 'Sum error over rows', 'rows', 'error')

# plot sum error over cols
plot(cols, errors_sum.mean(axis=0), 'Sum error over cols', 'cols', 'error')

# plot sumproduct error over rows
plot(rows, errors_sumproduct.mean(axis=1), 'Sum-product error over rows', 'rows', 'error')

# plot sumproduct error over cols
plot(cols, errors_sumproduct.mean(axis=0), 'Sum-product error over cols', 'cols', 'error')

# plot sum speedup over rows
plot(rows, speedups_sum.mean(axis=1), 'Sum speedup over rows', 'rows', 'speedup')

# plot sum speedup over cols
plot(cols, speedups_sum.mean(axis=0), 'Sum speedup over cols', 'cols', 'speedup')

# plot sumproduct speedup over rows
plot(rows, speedups_sumproduct.mean(axis=1), 'Sum-product speedup over rows', 'rows', 'speedup')

# plot sumproduct speedup over cols
plot(cols, speedups_sumproduct.mean(axis=0), 'Sum-product speedup over cols', 'cols', 'speedup')
