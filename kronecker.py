from typing import Tuple
import numpy as np

from svd import SVD


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
    return matrix.reshape((-1, shape_a[1], shape_a[0]), order='F').reshape(shape_a[1], -1)


def compute_shapes(shape: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    :param shape: shape of the matrix
    :return: shapes of the matrices A and B
    """
    m, n = shape
    # choose the heights of the matrices to balance the sizes (m1, m2)
    m1 = int(np.sqrt(m))
    m2 = None
    for _ in range(m1):
        if m % m1 == 0:
            m2 = m // m1
            break
        else:
            m1 -= 1
    # choose the widths of the matrices to balance the sizes (n1, n2)
    n1 = int(np.sqrt(n))
    n2 = None
    for _ in range(n1):
        if n % n1 == 0:
            n2 = n // n1
            break
        else:
            n1 -= 1

    if m2 is None or n2 is None:
        raise ValueError('Could not find suitable dimensions for the matrices.')

    return (m1, n1), (m2, n2)


def kronecker_decomposition(svd: SVD,
                            shape_a: Tuple[int, int],
                            shape_b: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param svd: SVD object
    :param shape_a: shape of matrix A
    :param shape_b: shape of matrix B
    :return: A: 2D matrix, B: 2D matrix
    """
    u_mat, s_vec, vh_mat = svd.U, svd.s, svd.VH
    v_mat = vh_mat.transpose()
    sqrt_s = np.sqrt(s_vec[0])
    a_mat = reshape(u_mat[:, 0], shape_a) * sqrt_s
    b_mat = reshape(v_mat[:, 0], shape_b) * sqrt_s
    return a_mat, b_mat
