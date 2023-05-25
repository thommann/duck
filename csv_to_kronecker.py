import argparse

import numpy as np

from kronecker import kronecker_decomposition


def csv_to_kronecker(input: str, output_a: str, output_b: str) -> None:
    matrix = np.loadtxt(input, delimiter=',')
    a_mat, b_mat = kronecker_decomposition(matrix)
    np.savetxt(output_a, a_mat, delimiter=',')
    np.savetxt(output_b, b_mat, delimiter=',')


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description='SVD')
    parser.add_argument('--input', type=str, required=True, help='Path to input CSV file')
    parser.add_argument('--output_a', type=str, required=True, help='Path to output CSV file for matrix A')
    parser.add_argument('--output_b', type=str, required=True, help='Path to output CSV file for matrix B')
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
    csv_to_kronecker(args['input'], args['output_a'], args['output_b'])
