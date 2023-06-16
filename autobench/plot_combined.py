import matplotlib.pyplot as plt
import numpy as np

from autobench.params import col_decompositions, ks, factors


def plot_error_vs_rank(nr_factors: int = 1):
    plt.figure()
    plt.title(f'Error vs. Rank ({"sum" if nr_factors == 1 else f"sumproduct: {nr_factors} columns"})')
    plt.xlabel('Rank')
    plt.xticks(ks)
    plt.ylabel('Error')
    plt.yscale('log')
    plt.grid(True)
    # Retrieve error from data
    for decomposition_type in col_decompositions:
        decomposition_suffix = "_cc" if decomposition_type == "cc" else "_sc" if decomposition_type == "sc" else ""
        color = 'red' if decomposition_type == 'cc' else 'green' if decomposition_type == 'sc' else 'blue'
        errors = []
        for rank in ks:
            filepath_orig = f'data/results/results_db_orig_factors{nr_factors}_rank{rank}{decomposition_suffix}.csv'
            filepath_kron = f'data/results/results_db_kron_factors{nr_factors}_rank{rank}{decomposition_suffix}.csv'
            results_orig = np.loadtxt(filepath_orig, delimiter=',')
            results_kron = np.loadtxt(filepath_kron, delimiter=',')
            relative_errors = np.abs(results_orig - results_kron) / np.abs(results_orig)
            mean_relative_error = np.mean(relative_errors)
            errors.append(mean_relative_error)
        plt.plot(ks, errors, color=color, label=decomposition_type)
    plt.legend()
    plt.show()


for nr_factors in factors:
    plot_error_vs_rank(nr_factors)
