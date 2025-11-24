#!/usr/bin/env python3
import numpy as np


if __name__ == "__main__":
    arr = np.load("data/actions_oldp.npy")
    print(arr[1090:1190])
    #print(np.where(arr[:,4] > 0.02))
