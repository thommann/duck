import numpy as np
import torch

from ML.params import middle_layer


def extract_parameters():
    state_dict = torch.load(f'data/iris-model{middle_layer[0]}x{middle_layer[1]}.pth')
    for key, value in state_dict.items():
        value = np.atleast_2d(value)
        for val in [value, value.T]:
            size = val.shape
            print(key, size)
            np.savetxt(f"data/{key}_{size[0]}x{size[1]}.csv", val, delimiter=",")


if __name__ == "__main__":
    extract_parameters()
