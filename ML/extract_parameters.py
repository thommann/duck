import numpy as np
import torch

from ML.params import middle_layer, fc_layers, max_k
from src.kronecker import svd, compute_shapes, kronecker_decomposition


def calculate_kronecker(matrix: np.ndarray, k=1, cc=False) -> \
        tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], tuple[int, int]]:
    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes(shape_c, compress_cols=cc)
    u, s, vh = svd(matrix, shape_a)
    a, b = kronecker_decomposition(u, s, vh, shape_a, shape_b, k=k)
    return a, b, shape_a, shape_b


def to_tensor(matrix: np.ndarray, rank: int = None) -> np.ndarray:
    nr_cols = 3 if rank is None else 4
    tensor = np.zeros((matrix.shape[0] * matrix.shape[1], nr_cols))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            tensor[i * matrix.shape[1] + j, :3] = [i, j, matrix[i, j]]
            if rank is not None:
                tensor[i * matrix.shape[1] + j, 3] = rank
    return tensor


def extract_parameters(model: str):
    state_dict = torch.load(f'data/{model}-model{middle_layer[0]}x{middle_layer[1]}.pth')
    for w, b in fc_layers:
        weight = np.atleast_2d(state_dict[w]).T
        bias = np.atleast_2d(state_dict[b])
        fc = np.vstack((weight, bias))
        a, b, _, _ = calculate_kronecker(fc, k=max_k, cc=False)
        fc_transpose = fc.T

        # Save to file in tensor notation
        fc_tensor = to_tensor(fc_transpose)
        a_tensors = []
        b_tensors = []
        for k in range(max_k):
            a_tensors.append(to_tensor(a[k], rank=k))
            b_tensors.append(to_tensor(b[k].T, rank=k))

        a_tensor = np.vstack(a_tensors)
        b_tensor = np.vstack(b_tensors)

        layer = w.replace('.weight', '')
        filename = f"{model}_{layer}_{middle_layer[0]}x{middle_layer[1]}"
        np.savetxt(f"data/{filename}.csv", fc_tensor, delimiter=',', header="row,col,val", comments='')
        np.savetxt(f"data/{filename}_a.csv", a_tensor, delimiter=',', header="row,col,val,k", comments='')
        np.savetxt(f"data/{filename}_b.csv", b_tensor, delimiter=',', header="row,col,val,k", comments='')


def main():
    extract_parameters("iris")
    # extract_parameters("mnist")


if __name__ == "__main__":
    main()
