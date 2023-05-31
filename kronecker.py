from typing import Tuple

import numpy as np


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    :param matrix: 2D matrix
    :return: 1D vector
    """
    return matrix.ravel(order='F')


def reshape(vector: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    :param vector: 1D vector
    :param shape:
    :return: 2D matrix
    """
    return vector.reshape(shape, order='F')


def massage(matrix: np.ndarray, shape_a: Tuple[int, int]) -> np.ndarray:
    """
    :param matrix: 2D matrix
    :param shape_a: shape of matrix A
    :return: 2D matrix
    """
    if not isinstance(shape_a, tuple) or len(shape_a) != 2:
        raise ValueError('shape_a must be a tuple of length 2.')

    return np.vstack(
        [vec(block) for col in np.split(matrix, shape_a[1], axis=1) for block in np.split(col, shape_a[0], 0)])


def compute_shapes(shape: Tuple[int, int], compress_cols: bool = True) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    :param shape: shape of the matrix
    :param compress_cols: if True, compress the columns of the matrix
    :return: shapes of the matrices A and B
    """
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError('shape must be a tuple of length 2.')

    m, n = shape
    # choose the heights of the matrices to balance the sizes (m1, m2)
    m1 = int(np.sqrt(m))
    for _ in range(m1):
        if m % m1 == 0:
            break
        else:
            m1 -= 1
    m2 = m // m1
    # choose the widths of the matrices to balance the sizes (n1, n2)
    if not compress_cols:
        n1 = n
    else:
        n1 = int(np.sqrt(n))
        for _ in range(n1):
            if n % n1 == 0:
                break
            else:
                n1 -= 1
    n2 = n // n1

    return (m1, n1), (m2, n2)


def kronecker_decomposition(matrix: np.ndarray, compress_cols=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param matrix: 2D matrix
    :param compress_cols: if True, compress the columns of the matrix
    :return: A: 2D matrix, B: 2D matrix
    """
    shape_a, shape_b = compute_shapes(matrix.shape, compress_cols=compress_cols)
    massaged_matrix = massage(matrix, shape_a)
    print("Computing SVD...")
    u_mat, s_vec, vh_mat = np.linalg.svd(massaged_matrix, full_matrices=False)
    print("Done.")
    v_mat = vh_mat.transpose()
    sqrt_s = np.sqrt(s_vec[0])
    a_mat = reshape(u_mat[:, 0], shape_a) * sqrt_s
    b_mat = reshape(v_mat[:, 0], shape_b) * sqrt_s
    return a_mat, b_mat
