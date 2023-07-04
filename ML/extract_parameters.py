import numpy as np
import torch

from ML.params import middle_layer


def extract_parameters(model: str):
    state_dict = torch.load(f'data/{model}-model{middle_layer[0]}x{middle_layer[1]}.pth')
    for key, value in state_dict.items():
        value = np.atleast_2d(value)
        for val in [value, value.T]:
            size = val.shape
            print(key, size)
            np.savetxt(f"data/{model}_{key}_{size[0]}x{size[1]}.csv", val, delimiter=",")


def main():
    extract_parameters("iris")
    extract_parameters("mnist")


if __name__ == "__main__":
    main()
