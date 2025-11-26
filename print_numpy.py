#!/usr/bin/env python3
import numpy as np


if __name__ == "__main__":
    arr = np.load("data/actions_newp.npy")
    print(arr[1038:1070])
    print(np.where(arr[:,4] > 0.02))
