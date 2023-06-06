import argparse

import numpy as np

from src.image_to_matrix import image_to_matrix
from src.kronecker_to_db import kronecker_to_db
from src.matrix_to_kronecker import matrix_to_kronecker, matrix_to_kronecker_columns


def try_image_to_matrix(input_path: str, matrix_c: str, dimensions: tuple[int, int]) -> tuple:
    try:
        matrix = np.loadtxt(matrix_c, delimiter=',')
        initialized = True
    except OSError:
        matrix = image_to_matrix(input_path, matrix_c, dimensions)
        initialized = False
    return matrix, initialized


def image_to_db(input_path: str, dimensions: tuple, name: str, k: int = 1,
                compress_cols: bool = False, single_column: bool = False) -> None:
    # We can't compress columns and decompose every column separately at the same time
    assert not (compress_cols and single_column)

    # 0. Set up paths
    full_name = f"{name}_{dimensions[0]}x{dimensions[1]}"
    matrix_c = f"data/matrices/{full_name}.csv"
    suffix = "_sc" if single_column else "_cc" if compress_cols else ""
    matrix_a = matrix_c.replace(".csv", f"{suffix}_a.csv")
    matrix_b = matrix_c.replace(".csv", f"{suffix}_b.csv")
    rank_append = '' if k == 1 else f"_rank_{k}"
    database = f"data/databases/{full_name}{suffix}{rank_append}.db"

    # 1. Create matrix C from image
    matrix, initialized = try_image_to_matrix(input_path, matrix_c, dimensions)

    # 2. Create matrices A and B from matrix C
    if single_column:
        # 2.a. Decompose every column separately
        matrix_to_kronecker_columns(matrix_c, matrix_a, matrix_b, k=k, matrix=matrix, initialized=initialized)
    else:
        # 2.b. Decompose the whole matrix
        matrix_to_kronecker(matrix_c, matrix_a, matrix_b, k=k, compress_cols=compress_cols, matrix=matrix,
                            initialized=initialized)

    # 3. Create database from matrices A, B and C
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
    parser.add_argument('--single_column', '-sc',
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help='Compress every column of the output matrix separately')
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

    # Dimensions must be positive integers
    if args['dimensions'][0] < 1 or args['dimensions'][1] < 1:
        raise ValueError('Dimensions must be positive integers')

    # Compress columns and single column can't be used at the same time
    if args['compress_cols'] and args['single_column']:
        raise ValueError('Compress columns and single column can\'t be used at the same time')

    # Rank must be a positive integer
    if args['rank'] < 1:
        raise ValueError('Rank must be a positive integer')

    return args


if __name__ == '__main__':
    args = parse_args()
    image_to_db(args['input'], args['dimensions'], args['name'],
                k=args['rank'], compress_cols=args['compress_cols'], single_column=args['single_column'])
