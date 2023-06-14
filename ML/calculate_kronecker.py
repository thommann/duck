import numpy as np

from src.kronecker import svd, compute_shapes, kronecker_decomposition


def calculate_kronecker(matrix, k=1) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(matrix.T).T
    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes(shape_c)
    u, s, vh = svd(matrix, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=k)
    return a, b


if __name__ == '__main__':
    matrices = ['fc1.weight_4x100', 'fc1.bias_100x1', 'fc2.weight_100x50', 'fc2.bias_50x1', 'fc3.weight_50x3',
                'fc3.bias_3x1']

    for matrix in matrices:
        filepath = f"{matrix}.csv"
        c = np.loadtxt(filepath, delimiter=',')
        a, b = calculate_kronecker(c, k=1)
        np.savetxt(f"{matrix}_a.csv", a, delimiter=',')
        np.savetxt(f"{matrix}_b.csv", b, delimiter=',')

    print("Done!")
