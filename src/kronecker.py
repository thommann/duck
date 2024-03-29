import numpy as np


def vec(matrix: np.ndarray) -> np.ndarray:
    """
    :param matrix: 2D matrix
    :return: 1D vector
    """
    return matrix.ravel(order='F')


def reshape(vector: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """
    :param vector: 1D vector
    :param shape:
    :return: 2D matrix
    """
    return vector.reshape(shape, order='F')


def massage(matrix: np.ndarray, shape_a: tuple[int, int]) -> np.ndarray:
    """
    :param matrix: 2D matrix
    :param shape_a: shape of matrix A
    :return: 2D matrix
    """
    if not isinstance(shape_a, tuple) or len(shape_a) != 2:
        raise ValueError('shape_a must be a tuple of length 2.')

    # matrix must be a 2D matrix
    if len(matrix.shape) != 2:
        raise ValueError('matrix must be a 2D matrix.')

    return np.vstack(
        [vec(block) for col in np.split(matrix, shape_a[1], axis=1) for block in np.split(col, shape_a[0], 0)])


def compute_shapes(shape: tuple[int, int], compress_cols: bool = False) -> tuple[tuple[int, int], tuple[int, int]]:
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
    if compress_cols:
        n1 = int(np.sqrt(n))
        for _ in range(n1):
            if n % n1 == 0:
                break
            else:
                n1 -= 1
    else:
        n1 = n
    n2 = n // n1

    return (m1, n1), (m2, n2)


def kronecker_decomposition(u_mat: np.ndarray, s_vec: np.ndarray, vh_mat: np.ndarray,
                            shape_a: tuple[int, int], shape_b: tuple[int, int],
                            k: int = 1) -> tuple[list[np.ndarray], list[np.ndarray]]:
    v_mat = vh_mat.transpose()
    scales = np.sqrt(s_vec)
    u_mat_scaled = u_mat * scales
    v_mat_scaled = v_mat * scales
    a_matrices = []
    b_matrices = []
    for i in range(k):
        if i < u_mat_scaled.shape[1]:
            a_matrices.append(reshape(u_mat_scaled[:, i], shape_a))
        else:
            a_matrices.append(np.zeros(shape_a))

        if i < v_mat_scaled.shape[1]:
            b_matrices.append(reshape(v_mat_scaled[:, i], shape_b))
        else:
            b_matrices.append(np.zeros(shape_b))

    return a_matrices, b_matrices


def massaged_svd(matrix: np.ndarray, shape_a: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    massaged_matrix = massage(matrix, shape_a)
    u_mat, s_vec, vh_mat = np.linalg.svd(massaged_matrix, full_matrices=False)
    return u_mat, s_vec, vh_mat
