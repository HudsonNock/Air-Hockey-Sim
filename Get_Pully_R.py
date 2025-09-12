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
import agent_processing_camera_delay as ap
import serial
import gc
import struct
import tensordict
from tensordict import TensorDict
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
        
def deq(q, xmin, xmax):
    y = q/32767
    return (y+1)/2 * (xmax - xmin) + xmin
        

def get_mallet(ser):    
    FMT = '<hhhhhhB'    # 6×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)
    
    # Read entire buffer
    while ser.in_waiting < 33:
        pass
    buffer = ser.read(ser.in_waiting)
    
    # Start from the last 29 bytes
    search_buffer = buffer[-33:]
    
    # Find the start marker (0xAA)
    start_idx = -1
    for i in range(len(search_buffer)-3):
        if search_buffer[i] == 0xFF and search_buffer[i+1] == 0xFF and search_buffer[i+2] == 0xFF:
            start_idx = i+2
            break
    
    if start_idx == -1:
        print("No start marker found")
        return None, None, None, False
    
    # Check if we have enough bytes for a complete frame after the start marker
    remaining_bytes = len(search_buffer) - start_idx - 1  # -1 for the start marker itself
    if remaining_bytes < FRAME_SIZE + 1:  # +1 for end marker
        print("Not enough bytes for complete frame")
        return None, None, None, False
    
    # Extract the frame data
    frame_start = start_idx + 1
    frame_end = frame_start + FRAME_SIZE
    raw = search_buffer[frame_start:frame_end]
    
    # Check end marker
    if frame_end >= len(search_buffer) or search_buffer[frame_end] != 0x55:
        print(buffer)
        print("Invalid end marker")
        return None, None, None, False
    
    # Unpack the data
    p0, p1, v0, v1, a0, a1, chk = struct.unpack(FMT, raw)

    # Verify checksum
    c = 0
    for b in raw[:-1]:
        c ^= b
    if c != chk:
        print("bad checksum", c, chk)
        return None, None, None, False
    
    pos = np.array([deq(p0, 0.03, 0.04), deq(p1, -1, 2)])
    vel = np.array([deq(v0, -30, 30), deq(v1, -30, 30)])
    acc = np.array([deq(a0, -150, 150), deq(a1, -150, 150)])
    
    return pos, vel, acc, True    
               


def main():
    PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
    BAUD = 460800

    # === CONNECT ===
    ser = serial.Serial(PORT, BAUD, timeout=0)
    
    time.sleep(1)
    
    ser.write(b'\n')
    while ser.in_waiting == 0:
        continue
    ser.read(ser.in_waiting)

    ser.write(b'\n')
    input("Place mallet bottom right")
    
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.read(ser.in_waiting)
        
    input("Place mallet bottom left")
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.read(ser.in_waiting)
    
    while True:
        passed = False
        while not passed:
            pos, vel, acc, passed = get_mallet(ser)
        print(pos[0])
        time.sleep(0.01)

if __name__ == "__main__":
    main()
