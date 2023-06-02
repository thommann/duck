import argparse

import numpy as np

from src.image_to_matrix import image_to_matrix
from src.kronecker_to_db import kronecker_to_db
from src.matrix_to_kronecker import matrix_to_kronecker


def image_to_db(input_path: str, dimensions: tuple, name: str, k: int = 1, compress_cols: bool = False) -> None:
    full_name = f"{name}_{dimensions[0]}x{dimensions[1]}"
    if compress_cols:
        full_name += "_cc"
    matrix_c = f"data/matrices/{full_name}.csv"
    matrix_a = matrix_c.replace(".csv", "_a.csv")
    matrix_b = matrix_c.replace(".csv", "_b.csv")
    rank_append = '' if k == 1 else f"_rank_{k}"
    database = f"data/databases/{full_name}{rank_append}.db"
    try:
        matrix = np.loadtxt(matrix_c, delimiter=',')
        initialized = True
    except OSError:
        matrix = image_to_matrix(input_path, matrix_c, dimensions)
        initialized = False
    matrix_to_kronecker(matrix_c, matrix_a, matrix_b, k=k, compress_cols=compress_cols, matrix=matrix,
                        initialized=initialized)
    kronecker_to_db(matrix_c, matrix_a, matrix_b, database)
    print("All done!", flush=True)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Image to csv converter")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input image")
    parser.add_argument("--name", "-n", type=str, required=True, help="Name of the matrices and db")
    parser.add_argument("--dimensions", "-d", type=int, nargs=2, required=True, help="Dimensions of the output matrix")
    parser.add_argument('--compress_cols', '-cc',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Compress columns of the output matrix')
    parser.add_argument('--rank', '-r', '-k', type=int, default=1, help='Rank of the kronecker product')
    args = vars(parser.parse_args())

    # Input mus be png or jpg
    if not args['input'].endswith('.png') and not args['input'].endswith('.jpg'):
        raise ValueError('Input must be a png or jpg file')

    # Name must be a valid identifier
    if not args['name'].isidentifier():
        raise ValueError('Name must be a valid identifier')

    # Dimensions must be a tuple of 2 integers
    if len(args['dimensions']) != 2:
        raise ValueError('Dimensions must be a tuple of 2 integers')

    # Rank must be a positive integer
    if args['rank'] < 1:
        raise ValueError('Rank must be a positive integer')

    return args


if __name__ == '__main__':
    args = parse_args()
    image_to_db(args['input'], args['dimensions'], args['name'], k=args['rank'], compress_cols=args['compress_cols'])
