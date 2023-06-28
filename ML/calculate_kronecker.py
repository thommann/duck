import numpy as np

from ML.params import middle_layer, k
from src.kronecker import svd, compute_shapes, kronecker_decomposition


def do_decomposition(matrix, k=1, cc=False) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.atleast_2d(matrix.T).T
    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes(shape_c, compress_cols=cc)
    u, s, vh = svd(matrix, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=k)
    return a, b


def calculate_kronecker():
    matrices = [f'fc1.weight_4x{middle_layer[0]}', f'fc1.bias_{middle_layer[0]}x1',
                f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}',
                f'fc2.bias_{middle_layer[1]}x1', f'fc3.weight_{middle_layer[1]}x3', f'fc3.bias_3x1']

    for matrix in matrices:
        filepath = f"data/{matrix}.csv"
        c = np.loadtxt(filepath, delimiter=',')
        a, b = do_decomposition(c, k=k)
        np.savetxt(f"data/{matrix}_a.csv", a, delimiter=',')
        np.savetxt(f"data/{matrix}_b.csv", b, delimiter=',')

    print("Done!")


if __name__ == "__main__":
    calculate_kronecker()
