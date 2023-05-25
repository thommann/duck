import pickle
import numpy as np


class SVD:
    U: np.ndarray   # 2D matrix
    s: np.ndarray   # 1D vector
    VH: np.ndarray  # 2D matrix

    def __init__(self, matrix: np.ndarray):
        self.U, self.s, self.VH = np.linalg.svd(matrix, full_matrices=False)


def matrix_to_svd(matrix: np.ndarray) -> SVD:
    return SVD(matrix)


def csv_to_matrix(file_path: str) -> np.ndarray:
    return np.genfromtxt(file_path, delimiter=',')


def svd_to_pickle(svd: SVD, file_path: str) -> None:
    with open(file_path, 'wb') as f:
        pickle.dump(svd, f)


def pickle_to_svd(file_path: str) -> SVD:
    with open(file_path, 'rb') as f:
        return pickle.load(f)
