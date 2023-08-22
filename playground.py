import timeit

import cv2
import numpy as np
import pandas as pd
import duckdb
from src.kronecker import kronecker_decomposition, massaged_svd


def get_mat(mat_dim, img_name='webb.png') -> np.ndarray:
    img_dir = 'data/images'
    img = cv2.imread(f'{img_dir}/{img_name}', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, mat_dim[::-1])
    mat = np.array(img)
    mat = mat / 255
    mat += np.random.normal(size=mat.shape) * 0.1
    mat = np.clip(mat, 0, 1)
    return mat


def compute_shapes(shape: tuple[int, int],
                   compression_type: str = "compress_individual") -> tuple[tuple[int, int], tuple[int, int]]:
    """
    :param shape: shape of the mat
    :param compression_type:
        "compress_individual": compress each col of the mat individually.
        "compress_fully": compress the mat, where #cols in A ~= #cols in B ~= sqrt(#cols in mat).
        "compress_left": compress the mat, where #cols in A = #cols in mat and #cols in B = 1.
        "compress_right": compress the mat, where #cols in A = 1 and #cols in B = #cols in mat.
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
    if compression_type == "compress_fully":
        n1 = int(np.sqrt(n))
        for _ in range(n1):
            if n % n1 == 0:
                break
            else:
                n1 -= 1
        n2 = n // n1
    elif compression_type == "compress_left":
        n1 = n
        n2 = 1
    elif compression_type == "compress_right":
        n1 = 1
        n2 = n
    elif compression_type == "compress_individual":
        n1 = 1
        n2 = 1
    else:
        raise ValueError(
            'compression_type must be one of "compress_individual", "compress_fully", "compress_left", '
            '"compress_right".')

    return (m1, n1), (m2, n2)


def compute_krone_decomposition(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat_dim = mat.shape
    a_shape, b_shape = compute_shapes(mat_dim)
    number_of_cols = mat_dim[1]
    rank = 2
    a_mat = np.zeros((a_shape[0], number_of_cols * rank))
    b_mat = np.zeros((b_shape[0], number_of_cols * rank))
    for col_idx in range(number_of_cols):
        col = np.atleast_2d(mat[:, col_idx]).T
        u, s, vh = massaged_svd(col, shape_a=a_shape)
        a_cols, b_cols = kronecker_decomposition(u, s, vh, a_shape, b_shape, k=rank)
        for k in range(rank):
            a_mat[:, col_idx + k * number_of_cols] = a_cols[k].squeeze()
            b_mat[:, col_idx + k * number_of_cols] = b_cols[k].squeeze()

    return a_mat, b_mat


def double_mat_df(mat_df: pd.DataFrame) -> pd.DataFrame:
    mat_df_1 = mat_df.copy()
    mat_df_1['x_1'] = 1
    mat_df_2 = mat_df.copy()
    mat_df_2['x_1'] = 2
    return pd.concat([mat_df_1, mat_df_2])


def compare_queries(original_query, krone_query, S, A, B):
    conn = duckdb.connect()

    original_result = conn.execute(original_query).fetchall()
    krone_result = conn.execute(krone_query).fetchall()
    print(f"Original result: {original_result}")
    print(f"Krone result: {krone_result}")

    original_time = timeit.timeit(lambda: conn.execute(original_query).fetchall(), number=100)
    krone_time = timeit.timeit(lambda: conn.execute(krone_query).fetchall(), number=100)
    print(f"Original time: {original_time:.2f}")
    print(f"Krone time: {krone_time:.2f}")


def compare_count(S, A, B):
    original_query = """
    SELECT x_1, COUNT(s_1)
    FROM S
    GROUP BY x_1
    """

    krone_query = """
    SELECT x_1, count_a * count_b
    FROM (
        SELECT x_1, COUNT(a_1) as count_a
        FROM A
        GROUP BY x_1
    ) NATURAL JOIN (
        SELECT x_1, COUNT(b_1) as count_b
        FROM B
        GROUP BY x_1
    )
    """

    compare_queries(original_query, krone_query, S, A, B)


def compare_sum(S, A, B):
    original_query = """
    SELECT x_1, SUM(s_1)
    FROM S
    GROUP BY x_1
    """

    krone_query = """
    SELECT x_1, sum_a * sum_b
    FROM (
        SELECT x_1, SUM(a_1) as sum_a
        FROM A
        GROUP BY x_1
    ) NATURAL JOIN (
        SELECT x_1, SUM(b_1) as sum_b
        FROM B
        GROUP BY x_1
    )
    """

    compare_queries(original_query, krone_query, S, A, B)


def compare_sumproduct(S, A, B):
    original_query = """
    SELECT x_1, SUM(s_1 * s_2)
    FROM S
    GROUP BY x_1
    """

    krone_query = """
    SELECT x_1, sumprod_a * sumprod_b
    FROM (
        SELECT x_1, SUM(a_1 * a_2) as sumprod_a
        FROM A
        GROUP BY x_1
    ) NATURAL JOIN (
        SELECT x_1, SUM(b_1 * b_2) as sumprod_b
        FROM B
        GROUP BY x_1
    )
    """

    compare_queries(original_query, krone_query, S, A, B)


def main():
    mat_dim = (100_000, 10)
    mat = get_mat(mat_dim)
    mat_df = pd.DataFrame(mat, columns=[f's_{i + 1}' for i in range(mat_dim[1])])
    S = double_mat_df(mat_df)
    # Compute the krone decomposition
    a_mat, b_mat = compute_krone_decomposition(mat)
    a_mat_df = pd.DataFrame(a_mat, columns=[f'a_{i + 1}' for i in range(a_mat.shape[1])])
    b_mat_df = pd.DataFrame(b_mat, columns=[f'b_{i + 1}' for i in range(b_mat.shape[1])])
    A = double_mat_df(a_mat_df)
    B = double_mat_df(b_mat_df)

    compare_count(S, A, B)
    compare_sum(S, A, B)
    compare_sumproduct(S, A, B)


if __name__ == '__main__':
    main()
