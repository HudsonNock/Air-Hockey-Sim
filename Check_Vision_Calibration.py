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


table_bounds = np.array([1.9885, 0.995])

margin = 0.065
margin_bottom = 0.1

mallet_r = 0.0508
margin_bounds = 0.0
mallet_bounds = np.array([[margin_bounds + mallet_r, table_bounds[0]/2  + mallet_r/2], [margin_bounds+mallet_r, table_bounds[1]-margin_bounds-mallet_r]])



def main():
    img_shape = (1536, 1296)

    img = np.load('img_data_3.npy')
    
    pxls = np.load('pxls_data_3.npy')
    locations = np.load('location_data_3.npy')
    
    for pxl in pxls:
        cv2.circle(img, (int(pxl[0]) - 376, int(pxl[1])), 10, (255, 0, 0), -1)

    cv2.imshow('v1', img[::2, ::2])
    cv2.waitKey(0)
    
    
    
    #pxls = np.concatenate([pxls[:3], pxls[4:]], axis=0)
    #locations = np.concatenate([locations[:3], locations[4:]], axis=0)
    
    setup = tracker.SetupCamera()
    setup.run_extrinsics(img)
    
    track = tracker.CameraTracker(setup.rotation_matrix,
                                  setup.translation_vector,
                                  setup.z_pixel_map,
                                  70.44*10**(-3)) #(120.94)*10**(-3))
     
    err = []
     
    for i, pxl in enumerate(pxls):
        loc = extrinsic.global_coordinate_zpixel(pxl,
            track.rotation_matrix,
            track.translation_vector,
            track.intrinsic_matrix,
            track.distortion_coeffs,
            track.puck_z,
            track.z_pixel_map)[0:2]
            
        table_width = 0.9905
        table_height = 1.9885 
        
        #print("---")
            
        loc = np.array([table_height - loc[0], table_width - loc[1]])
        
        table_bounds = np.array([1.993, 0.992])
        loc2 = np.array([table_bounds[0] - locations[i,0], table_bounds[1]-locations[i,1]])
        err.append(np.linalg.norm(loc2 - loc))
        
        #print(loc)
        #print(loc2)
            
    print(err)
    print(len(err))
    print(np.mean(np.array(err)))

if __name__ == "__main__":
    main()
