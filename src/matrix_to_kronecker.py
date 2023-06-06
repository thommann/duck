import argparse

import numpy as np

from src.kronecker import kronecker_decomposition, svd, compute_shapes


def try_svd(prefix: str, suffix: str, initialized: bool, matrix: np.ndarray, shape_a: tuple[int, int]):
    try:
        if not initialized:
            raise OSError
        u_mat = np.loadtxt(f"{prefix}U{suffix}", delimiter=',')
        s_vec = np.loadtxt(f"{prefix}S{suffix}", delimiter=',')
        vh_mat = np.loadtxt(f"{prefix}VH{suffix}", delimiter=',')
        initialized = True
    except OSError:
        u_mat, s_vec, vh_mat = svd(matrix, shape_a)
        np.savetxt(f"{prefix}U{suffix}", u_mat, delimiter=',')
        np.savetxt(f"{prefix}S{suffix}", s_vec, delimiter=',')
        np.savetxt(f"{prefix}VH{suffix}", vh_mat, delimiter=',')
        initialized = False
    return u_mat, s_vec, vh_mat, initialized


def try_kronecker_decomposition(output_a: str, output_b: str,
                                initialized: bool,
                                u_mat: np.ndarray, s_vec: np.ndarray, vh_mat: np.ndarray,
                                shape_a: tuple[int, int], shape_b: tuple[int, int], k: int):
    try:
        if not initialized:
            raise OSError
        a_mat = np.loadtxt(output_a, delimiter=',')
        b_mat = np.loadtxt(output_b, delimiter=',')
    except OSError:
        a_mat, b_mat = kronecker_decomposition(u_mat, s_vec, vh_mat, shape_a, shape_b, k=k)
        np.savetxt(output_a, a_mat, delimiter=',')
        np.savetxt(output_b, b_mat, delimiter=',')
    return a_mat, b_mat


def matrix_to_kronecker(input_c: str,
                        output_a: str,
                        output_b: str,
                        k: int = 1,
                        compress_cols: bool = False,
                        matrix: np.ndarray | None = None,
                        initialized: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if matrix is None:
        matrix = np.loadtxt(input_c, delimiter=',', ndmin=2)
    shape_c = matrix.shape

    shape_a, shape_b = compute_shapes(shape_c, compress_cols=compress_cols)

    if compress_cols:
        np.savetxt(input_c.replace("/matrices/", "/bcols/"), [shape_b[1]], delimiter=',')

    cc = '_cc' if compress_cols else ''
    prefix, suffix = 'data/svd/', f'_{shape_c[0]}x{shape_c[1]}{cc}.csv'

    u_mat, s_vec, vh_mat, initialized = try_svd(prefix, suffix, initialized, matrix, shape_a)

    a_mat, b_mat = try_kronecker_decomposition(output_a, output_b, initialized, u_mat, s_vec, vh_mat, shape_a, shape_b,
                                               k)

    return a_mat, b_mat


def matrix_to_kronecker_columns(input_c: str,
                                output_a: str,
                                output_b: str,
                                k: int = 1,
                                matrix: np.ndarray | None = None,
                                initialized: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if matrix is None:
        matrix = np.loadtxt(input_c, delimiter=',', ndmin=2)

    shape_c = matrix.shape
    shape_a, shape_b = compute_shapes((shape_c[0], 1))
    assert shape_a[1] == 1
    assert shape_b[1] == 1

    prefix = 'data/svd/'

    a_mat = np.zeros((shape_a[0], matrix.shape[1]))
    b_mat = np.zeros((shape_b[0], matrix.shape[1]))
    for col in range(matrix.shape[1]):
        suffix = f'_{shape_c[0]}x{shape_c[1]}_col_{col}.csv'
        # 1. SVD
        u_mat, s_vec, vh_mat, initialized = try_svd(prefix, suffix, initialized, np.atleast_2d(matrix[:, col]).T,
                                                    shape_a)
        # 2. Kronecker decomposition
        a_col, b_col = try_kronecker_decomposition(output_a, output_b, initialized, u_mat, s_vec, vh_mat, shape_a,
                                                   shape_b, k)
        assert a_col.shape[1] == 1
        assert b_col.shape[1] == 1
        a_mat[:, col] = a_col
        b_mat[:, col] = b_col

    return a_mat, b_mat


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Convert a matrix to Kronecker form and save the result to CSV files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('-oa', '--output_a', type=str, required=True, help='Path to output CSV file for matrix A')
    parser.add_argument('-ob', '--output_b', type=str, required=True, help='Path to output CSV file for matrix B')
    parser.add_argument('--compress_cols', '-cc',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Compress columns of matrix A')
    parser.add_argument('--single_column', '-sc',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Decompose each column of the input matrix separately')
    parser.add_argument('--rank', '-r', '-k', type=int, default=1, help='Rank of the Kronecker decomposition')
    args = vars(parser.parse_args())

    # Input must be a CSV file
    if not args['input'].endswith('.csv'):
        raise ValueError('Input must be a CSV file')

    # Output must be a CSV file
    if not args['output_a'].endswith('.csv'):
        raise ValueError('Output A must be a CSV file')
    if not args['output_b'].endswith('.csv'):
        raise ValueError('Output B must be a CSV file')

    # Compress columns and single column cannot be used together
    if args['compress_cols'] and args['single_column']:
        raise ValueError('Compress columns and single column cannot be used together')

    # Rank must be positive
    if args['rank'] < 1:
        raise ValueError('Rank must be positive')

    return args


if __name__ == '__main__':
    args = parse_args()
    if args['single_column']:
        matrix_to_kronecker_columns(args['input'], args['output_a'], args['output_b'], k=args['rank'])
    else:
        matrix_to_kronecker(args['input'], args['output_a'], args['output_b'], k=args['rank'],
                            compress_cols=args['compress_cols'])
