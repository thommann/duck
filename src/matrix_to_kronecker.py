import argparse

import numpy as np

from src.kronecker import kronecker_decomposition, svd, compute_shapes


def matrix_to_kronecker(input_c: str, output_a: str, output_b: str, k: int = 1, compress_cols: bool = False) -> None:
    matrix = np.loadtxt(input_c, delimiter=',', ndmin=2)
    shape_c = matrix.shape

    shape_a, shape_b = compute_shapes(shape_c, compress_cols=compress_cols)
    u_mat, s_vec, vh_mat = svd(matrix, shape_a)

    cc = '_cc' if compress_cols else ''
    prefix, suffix = 'data/svd/', f'_{shape_c[0]}x{shape_c[1]}{cc}.csv'
    np.savetxt(f"{prefix}U{suffix}", u_mat, delimiter=',')
    np.savetxt(f"{prefix}S{suffix}", s_vec, delimiter=',')
    np.savetxt(f"{prefix}VH{suffix}", vh_mat, delimiter=',')

    a_matrices, b_matrices = kronecker_decomposition(u_mat, s_vec, vh_mat, shape_a, shape_b, k=k)

    np.savetxt(output_a, a_matrices, delimiter=',')
    np.savetxt(output_b, b_matrices, delimiter=',')


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Convert a matrix to Kronecker form and save the result to CSV files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('-oa', '--output_a', type=str, required=True, help='Path to output CSV file for matrix A')
    parser.add_argument('-ob', '--output_b', type=str, required=True, help='Path to output CSV file for matrix B')
    parser.add_argument('--compress_cols', '-cc',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Compress columns of matrix A')
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

    # Rank must be positive
    if args['rank'] < 1:
        raise ValueError('Rank must be positive')

    return args


if __name__ == '__main__':
    args = parse_args()
    matrix_to_kronecker(args['input'], args['output_a'], args['output_b'], k=args['rank'],
                        compress_cols=args['compress_cols'])
