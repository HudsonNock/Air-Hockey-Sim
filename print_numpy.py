#!/usr/bin/env python3
import numpy as np


if __name__ == "__main__":
    arr = np.load("data/actions_overhead.npy")
    #arr = arr[:386]
    #np.save("data/actions_overhead.npy", arr)
    print(arr[:,4].cumsum()[-1])
    #print(np.where(arr[:,4] > 0.02))
