#!/usr/bin/env python3
import PySpin
import numpy as np
import time
import threading
import queue
import os
import sys
import ctypes
import ctypes.util
import tracker
import cv2
import serial
import gc
import struct
import tensordict
from tensordict import TensorDict
import torch
import extrinsic
import matplotlib.pyplot as plt


table_bounds = np.array([2.362, 1.144])

margin = 0.065
margin_bottom = 0.1

mallet_r = 0.0508
margin_bounds = 0.0
mallet_bounds = np.array([[margin_bounds + mallet_r, table_bounds[0]/2  + mallet_r/2], [margin_bounds+mallet_r, table_bounds[1]-margin_bounds-mallet_r]])

err = []

def main():
    for j in range(4,5):
        img_shape = (1536, 2048)

        img = np.load(f"new_data/img_data_{j}.npy")
        img = np.repeat(img[:,:,None], 3, axis=2)
        print(img.shape)
        
        pxls = np.load(f"new_data/pxls_data_{j}.npy")
        #pxls[:,0] = pxls[:,0] - 376
        
        locations = np.load(f"new_data/location_data_{j}.npy")
        print(len(pxls))
        
        #np.save(f"new_data/pxls_data_{j}.npy", pxls)
        
        for pxl in pxls:
            cv2.circle(img, (int(pxl[0]), int(pxl[1])), 10, (255, 0, 255), -1)

        
        
        #print(pxls)

        cv2.imshow('v1', img[::2, ::2])
        cv2.waitKey(0)
        
        
        
        
        #pxls = np.concatenate([pxls[:3], pxls[4:]], axis=0)
        #locations = np.concatenate([locations[:3], locations[4:]], axis=0)
        
        setup = tracker.SetupCamera()
        setup.run_extrinsics(img)
        
        track = tracker.CameraTracker(setup.rotation_matrix,
                                      setup.translation_vector,
                                      setup.z_pixel_map,
                                      np.zeros(img_shape, dtype=np.uint8),
                                      img_shape,
                                      (0,0))
         
        for i, pxl in enumerate(pxls):
            loc = extrinsic.global_coordinate_zpixel(pxl,
                track.rotation_matrix,
                track.translation_vector,
                track.intrinsic_matrix,
                track.distortion_coeffs,
                track.puck_z,
                track.z_pixel_map)[0:2]
            
            err.append([(locations[i,0] - loc[0]) * 1e3, (locations[i,1]-loc[1]) * 1e3])
            print("--")
            print(locations[0])
            print(loc[0])
            print(locations[1])
            print(loc[1])
            
            #print(loc)
            #print(loc2)
         
    
     
    np.save("new_data/calibration_err_test.npy", np.array(err)) 
    print(np.array(err))
    print(np.array(err).shape)

    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
    counts, edges = np.histogram(err, bins=bins)
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']

    print("\nError Distribution:")
    print("-" * 50)
    total = len(err)
    for label, count in zip(labels, counts):
        percentage = 100 * count / total if total > 0 else 0
        print(f"{label:>6} mm: {count:>5} ({percentage:>5.1f}%)")
        
def plot_err():
    err = np.load("new_data/calibration_err_test.npy")
    
    err = np.linalg.norm(err, axis=1)
    bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, np.inf]
    counts, edges = np.histogram(err, bins=bins)
    labels = ['0-1', '1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8', '8-9', '9-10', '10+']

    print("\nError Distribution:")
    print("-" * 50)
    total = len(err)
    for label, count in zip(labels, counts):
        percentage = 100 * count / total if total > 0 else 0
        print(f"{label:>6} mm: {count:>5} ({percentage:>5.1f}%)")
    

    # 2. Define the bins
    # We go from 0 to 3 with a step of 0.25
    bin_size = 0.25
    bins = np.arange(0, 3 + bin_size, bin_size)

    # 3. Create the plot
    plt.figure(figsize=(10, 6))
    plt.hist(err, bins=bins, edgecolor='black', color='skyblue', alpha=0.7)

    # 4. Formatting
    plt.title(f'Distribution of Points (Bin Size: {bin_size})', fontsize=14)
    plt.xlabel('Value Range', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.xticks(bins)  # This ensures the X-axis labels match your bin edges
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.savefig('new_data/err_distribution.png')
    

if __name__ == "__main__":
    plot_err()
    #main()
