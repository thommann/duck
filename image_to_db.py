import argparse

from image_to_matrix import image_to_matrix
from kronecker_to_db import kronecker_to_db
from matrix_to_kronecker import matrix_to_kronecker


def image_to_db(input_path: str, dimensions: tuple, name: str, rank: int = 1, compress_cols: bool = True) -> None:
    full_name = f"{name}_{dimensions[0]}x{dimensions[1]}"
    original = f"data/matrices/{full_name}.csv"
    matrix_a = f"{original[:-4]}_a.csv"
    matrix_b = f"{original[:-4]}_b.csv"
    database = f"data/databases/{full_name}.db"
    image_to_matrix(input_path, original, dimensions)
    matrix_to_kronecker(original, matrix_a, matrix_b, rank=rank, compress_cols=compress_cols)
    kronecker_to_db(original, matrix_a, matrix_b, database, rank=rank)


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Image to csv converter")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--name", type=str, required=True, help="Name of the matrices and db")
    parser.add_argument("--dimensions", type=int, nargs=2, required=True, help="Dimensions of the output matrix")
    parser.add_argument('--compress_cols',
                        action=argparse.BooleanOptionalAction,
                        default=True,
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
    image_to_db(args['input'], args['dimensions'], args['name'], rank=args['rank'], compress_cols=args['compress_cols'])
