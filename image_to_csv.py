import argparse

import cv2
import numpy as np


def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Image to csv converter")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=False,
        default=None,
        help="Path to the output csv file",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs=2,
        required=True,
        help="Dimensions of the output matrix",
    )

    args = vars(parser.parse_args())

    # Input mus be png or jpg
    if args['input'][-4:] not in ['.png', '.jpg']:
        raise ValueError('Input must be a png or jpg file')

    # If output is not specified, it will be the same as input
    if args['output'] is None:
        args['output'] = args['input'][:-4] + '.csv'
    # Output must be a csv file
    elif args['output'][-4:] != '.csv':
        raise ValueError('Output must be a csv file')

    # Dimensions must be a tuple of 2 integers
    if len(args['dimensions']) != 2:
        raise ValueError('Dimensions must be a tuple of 2 integers')

    return args


if __name__ == '__main__':
    args = parse_args()
    input_path = args['input']
    output_path = args['output']
    dimensions = args['dimensions']
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (dimensions[1], dimensions[0]))
    matrix = np.array(img, dtype=np.float32)
    matrix += np.random.random(matrix.shape)
    np.savetxt(output_path, matrix, delimiter=',')
