import numpy as np
import torch

from ML.params import middle_layer, fc_layers
from src.kronecker import svd, compute_shapes, kronecker_decomposition


def calculate_kronecker(matrix, k=1, cc=False) -> tuple[np.ndarray, np.ndarray]:
    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes(shape_c, compress_cols=cc)
    u, s, vh = svd(matrix, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=k)
    return a, b


def to_tensor(matrix: np.ndarray) -> np.ndarray:
    tensor = np.zeros((matrix.shape[0] * matrix.shape[1], 3))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            tensor[i * matrix.shape[1] + j] = [i, j, matrix[i, j]]
    return tensor


def extract_parameters(model: str):
    state_dict = torch.load(f'data/{model}-model{middle_layer[0]}x{middle_layer[1]}.pth')
    for w, b in fc_layers:
        weight = np.atleast_2d(state_dict[w]).T
        bias = np.atleast_2d(state_dict[b])
        fc = np.vstack((weight, bias))
        fc_transpose = fc.T
        a, b = calculate_kronecker(fc_transpose)
        a_transpose = a.T

        # Save to file in tensor notation
        fc_tensor = to_tensor(fc_transpose)
        a_tensor = to_tensor(a_transpose)
        b_tensor = to_tensor(b)

        layer = w.replace('.weight', '')
        filename = f"{model}_{layer}_{middle_layer[0]}x{middle_layer[1]}"
        np.savetxt(f"data/{filename}.csv", fc_tensor, delimiter=',', header="row,col,val", comments='')
        np.savetxt(f"data/{filename}_a.csv", a_tensor, delimiter=',', header="row,col,val", comments='')
        np.savetxt(f"data/{filename}_b.csv", b_tensor, delimiter=',', header="row,col,val", comments='')


def main():
    extract_parameters("iris")
    # extract_parameters("mnist")


if __name__ == "__main__":
    main()
