import argparse
from typing import Tuple

from kronecker import massage
from svd import csv_to_matrix, matrix_to_svd, svd_to_pickle


def parse_args() -> Tuple[str, str, Tuple[int, int]]:
    parser = argparse.ArgumentParser(description='SVD')
    parser.add_argument('--input', type=str, default='data.csv', help='input file path')
    parser.add_argument('--output', type=str, default='', help='output file path')
    parser.add_argument('--shape_a', type=int, nargs=2, help='shape of matrix A')
    args = vars(parser.parse_args())

    input = args['input']
    if input[-4:] != '.csv':
        raise ValueError('Input file must be a csv file')

    output = args['output']
    if output and output[-4:] != '.pkl':
        raise ValueError('Output file must be a pickle file')
    elif not output:
        args['output'] = input[:-4] + '.svd.pkl'

    shape_a = args['shape_a']

    return input, output, shape_a


if __name__ == '__main__':
    input, output, shape_a = parse_args()

    matrix = csv_to_matrix(input)
    massaged_matrix = massage(matrix, shape_a)
    svd = matrix_to_svd(massaged_matrix)
    svd_to_pickle(svd, output)
