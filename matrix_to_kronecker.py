import argparse

import numpy as np

from kronecker import kronecker_decomposition


def matrix_to_kronecker(input: str, output_a: str, output_b: str, compress_cols: bool = True) -> None:
    matrix = np.loadtxt(input, delimiter=',', ndmin=2)

    a_mat, b_mat = kronecker_decomposition(matrix, compress_cols=compress_cols)
    np.savetxt(output_a, a_mat, delimiter=',')
    np.savetxt(output_b, b_mat, delimiter=',')


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='Convert a matrix to Kronecker form and save the result to CSV files')
    parser.add_argument('-i', '--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('-oa', '--output_a', type=str, required=True, help='Path to output CSV file for matrix A')
    parser.add_argument('-ob', '--output_b', type=str, required=True, help='Path to output CSV file for matrix B')
    parser.add_argument('--compress_cols',
                        action=argparse.BooleanOptionalAction,
                        default=True,
                        help='Compress columns of matrix A')
    args = vars(parser.parse_args())

    # Input must be a CSV file
    if not args['input'].endswith('.csv'):
        raise ValueError('Input must be a CSV file')

    # Output must be a CSV file
    if not args['output_a'].endswith('.csv'):
        raise ValueError('Output A must be a CSV file')
    if not args['output_b'].endswith('.csv'):
        raise ValueError('Output B must be a CSV file')

    return args


if __name__ == '__main__':
    args = parse_args()
    matrix_to_kronecker(args['input'], args['output_a'], args['output_b'], compress_cols=args['compress_cols'])
