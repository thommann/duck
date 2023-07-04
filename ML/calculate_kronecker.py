import numpy as np

from ML.params import max_k, iris_weight_matrices, mnist_weight_matrices
from src.kronecker import svd, compute_shapes, kronecker_decomposition


def do_decomposition(matrix, k=1, cc=False) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(matrix.T).T
    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes(shape_c, compress_cols=cc)
    u, s, vh = svd(matrix, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=k)
    return a, b


def calculate_kronecker(model: str):
    weight_matrices = iris_weight_matrices if model == "iris" else mnist_weight_matrices if model == "mnist" else None
    if weight_matrices is None:
        raise ValueError(f"Invalid model name: {model}")
    for matrix in weight_matrices:
        filepath = f"data/{model}_{matrix}.csv"
        c = np.loadtxt(filepath, delimiter=',')
        a, b = do_decomposition(c, k=max_k)
        np.savetxt(f"data/{model}_{matrix}_a.csv", a, delimiter=',')
        np.savetxt(f"data/{model}_{matrix}_b.csv", b, delimiter=',')

    print("Done!")


def main():
    calculate_kronecker("iris")
    calculate_kronecker("mnist")


if __name__ == "__main__":
    main()
