#!/usr/bin/env python3
import numpy as np


if __name__ == "__main__":
    arr = np.load("actions_newp.npy")
    print(arr[1020:1120])
    #print(np.where(arr[:,4] > 0.02))
