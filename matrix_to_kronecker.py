import argparse

import numpy as np

from kronecker import kronecker_decomposition


def matrix_to_kronecker(input: str, output_a: str, output_b: str, k: int = 1, compress_cols: bool = False) -> None:
    matrix = np.loadtxt(input, delimiter=',', ndmin=2)

    a_matrices, b_matrices = kronecker_decomposition(matrix, rank=k, compress_cols=compress_cols)
    if k == 1:
        np.savetxt(output_a, a_matrices[0], delimiter=',')
        np.savetxt(output_b, b_matrices[0], delimiter=',')
    else:
        for r in range(k):
            rank = r + 1
            np.savetxt(output_a.replace('.csv', f'_{rank}.csv'), a_matrices[r], delimiter=',')
            np.savetxt(output_b.replace('.csv', f'_{rank}.csv'), b_matrices[r], delimiter=',')


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
