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

    layers = [[f'fc1.weight_4x{middle_layer[0]}', f'fc1.bias_1x{middle_layer[0]}'],
              [f'fc2.weight_{middle_layer[0]}x{middle_layer[1]}', f'fc2.bias_1x{middle_layer[1]}'],
              [f'fc3.weight_{middle_layer[1]}x3', f'fc3.bias_1x3']]

    for layer in layers:
        table_name = layer[0].replace(".weight", "")
        weights_file = f"data/{layer[0]}.csv"
        bias_file = f"data/{layer[1]}.csv"

        # Konecker decomposition
        weights = np.loadtxt(weights_file, delimiter=',')  # m x n
        bias = np.loadtxt(bias_file, delimiter=',')  # 1 x n

        combined = np.vstack((weights, bias))  # (m+1) x n
        a, b = do_decomposition(combined, k=k)  # a: (m+1) x k, b: k x n
        a_file = f"data/{table_name}_a.csv"
        b_file = f"data/{table_name}_b.csv"
        np.savetxt(a_file, a, delimiter=',')
        np.savetxt(b_file, b, delimiter=',')

    print("Done!")


if __name__ == "__main__":
    calculate_kronecker()
