import argparse

import cv2
import numpy as np


def image_to_matrix(input_path: str, output_path: str, dimensions: tuple) -> None:
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (dimensions[1], dimensions[0]))

    matrix = np.array(img, dtype=np.float32)
    matrix += np.random.random(matrix.shape)
    np.savetxt(output_path, matrix, delimiter=',')


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Image to csv converter")
    parser.add_argument("--input", type=str, required=True, help="Path to the input image")
    parser.add_argument("--output", type=str, required=True, help="Path to the output csv file")
    parser.add_argument("--dimensions", type=int, nargs=2, required=True, help="Dimensions of the output matrix")
    args = vars(parser.parse_args())

    # Input mus be png or jpg
    if not args['input'].endswith('.png') and not args['input'].endswith('.jpg'):
        raise ValueError('Input must be a png or jpg file')

    # Output must be a csv file
    if not args['output'].endswith('.csv'):
        raise ValueError('Output must be a csv file')

    # Dimensions must be a tuple of 2 integers
    if len(args['dimensions']) != 2:
        raise ValueError('Dimensions must be a tuple of 2 integers')

    return args


if __name__ == '__main__':
    args = parse_args()
    image_to_matrix(args['input'], args['output'], args['dimensions'])
