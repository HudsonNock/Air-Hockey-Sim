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
import agent_processing_no_NN as ap
import argparse

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

table_bounds = np.array([2.362, 1.144])

margin = 0.03
margin_bottom = 0.03

mallet_r = 0.1011 / 2

num_points = 11

Vmax = 24 * 0.8


class CircularMalletBuffer:
    def __init__(self, length: int):
        """
        Initialize a circular buffer for storing 3-float entries.
        
        Args:
            length (int): Maximum number of entries in the buffer.
        """
        self.length = length
        self.buffer = np.zeros((length, 3))
        self.index = 0

    def add(self, entry):
        """Add one entry (3 floats) to the buffer."""
        self.buffer[self.index] = entry
        self.index = (self.index + 1) % self.length

    def read(self) -> np.ndarray:
        """
        Return all buffer contents in chronological order.
        
        Returns:
            np.ndarray: Array of shape (N, 3), where N ≤ length.
        """
        return np.vstack((self.buffer[self.index:], self.buffer[:self.index]))

mallet_buffer = CircularMalletBuffer(11)
stored_buffer = bytearray()

def set_realtime_priority():
    os.sched_setaffinity(0, {2,3})
    SCHED_FIFO = 1
    sched_param = ctypes.c_int(99)
    libc = ctypes.CDLL(ctypes.util.find_library('c'))
    result = libc.sched_setscheduler(0, SCHED_FIFO, ctypes.byref(sched_param))
    
    if result == 0:
        print("Real-time priority set successfully")
    else:
        print("Failed to set real-time priority - run with sudo")
        
def check_isolation_status():
    """Check if CPU isolation is working properly"""
    print("\n=== CPU Isolation Status ===")
    
    # Check isolated CPUs
    try:
        with open('/sys/devices/system/cpu/isolated', 'r') as f:
            isolated = f.read().strip()
            print(f"Isolated CPUs: {isolated}")
    except:
        print("No CPUs isolated")
    
    # Check current process CPU affinity
    current_affinity = os.sched_getaffinity(0)
    print(f"Process CPU affinity: {current_affinity}")
    
    # Check current CPU being used
    try:
        with open('/proc/self/stat', 'r') as f:
            stat_data = f.read().split()
            current_cpu = stat_data[38]  # processor field
            print(f"Currently running on CPU: {current_cpu}")
    except:
        print("Could not determine current CPU")
    
    # Check other processes on our CPU
    try:
        pid = os.getpid()
        result = os.popen(f'ps -eo pid,comm,psr | grep " 2$"').read()
        print(f"Processes on CPU 2:")
        print(result)
    except:
        print("Could not check processes on CPU 2")
        
def deq(q, xmin, xmax):
    y = q/32767
    return (y+1)/2 * (xmax - xmin) + xmin
        

def get_mallet(ser):   
    global stored_buffer
    global mallet_buffer 
    FMT = '<hhhB'    # 3×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT) #11
    
    # Read entire buffer
    while ser.in_waiting < 11*2-1:
        pass

    buffer = ser.read(ser.in_waiting)
    
    if len(buffer) > (num_points+1) * 11 - 1:
        stored_buffer = bytearray(buffer[-((num_points+1)*11-1):])
    else:
        stored_buffer.extend(buffer)
    
    while len(stored_buffer) >= 11:
        # Find the start marker (0xAA)
        start_idx = -1
        for i in range(len(stored_buffer)-3):
            if stored_buffer[i] == 0xFF and stored_buffer[i+1] == 0xFF and stored_buffer[i+2] == 0xFF:
                start_idx = i+2
                break
        
        if start_idx == -1:
            break
        
        # Check if we have enough bytes for a complete frame after the start marker
        remaining_bytes = len(stored_buffer) - start_idx - 1  # -1 for the start marker itself
        if remaining_bytes < FRAME_SIZE + 1:  # +1 for end marker
            break
        
        # Extract the frame data
        frame_start = start_idx + 1
        frame_end = frame_start + FRAME_SIZE
        raw = stored_buffer[frame_start:frame_end]
        
        # Check end marker
        if stored_buffer[frame_end] != 0x55:
            print("frame end err")
            print(stored_buffer)
            print(1/0)
            break
        
        # Unpack the data
        p0, p1, dt1, chk = struct.unpack(FMT, raw)

        # Verify checksum
        c = 0
        for b in raw[:-1]:
            c ^= b
        if c != chk:
            print("bad checksum", c, chk)
            print(stored_buffer)
            print(1/0)
        else:
            entry = np.array([deq(p0, -1, 2), deq(p1, -1, 2), deq(dt1, 0, 3) / 1000.0])
            mallet_buffer.add(entry)
        
        if frame_end+1 < len(stored_buffer):
            del stored_buffer[:frame_end+1]
        else:
            break
            
def get_init_conditions(pred=0):
    global mallet_buffer
    mallet_data = mallet_buffer.read()
    ts = np.cumsum(mallet_data[:,2])
    coef_x = np.polyfit(ts, mallet_data[:,0], 2)
    coef_y = np.polyfit(ts, mallet_data[:,1], 2)
    
    t = ts[-1] + pred
    
    pos = np.array([coef_x[0]*t**2 + coef_x[1]*t + coef_x[2], \
                    coef_y[0]*t**2 + coef_y[1]*t + coef_y[2]])
                    
    vel = np.array([2*coef_x[0]*t + coef_x[1], \
                    2*coef_y[0]*t + coef_y[1]])
                    
    acc = np.array([2*coef_x[0], \
                    2*coef_y[0]])
                    
    return pos, vel, acc
    

def system_loop():
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement
    PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
    BAUD = 460800

    # === CONNECT ===
    ser = serial.Serial(PORT, BAUD, timeout=0)
    
    ser.write(b'\n')
    print("A")
    while ser.in_waiting == 0:
        continue
    print("B")
    ser.reset_input_buffer()

    ser.write(b'\n')
    input("Place mallet bottom right")
    
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.reset_input_buffer()
    
    input("Place mallet bottom left")
    
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
        
    time.sleep(0.1)
        
    pully_R = float(ser.readline().decode('utf-8').strip())
    print(f"Pulley radius measured as: {pully_R}")
    ap.pullyR = pully_R
    ap.C1 = [ap.Vmax * ap.pullyR / 2, ap.Vmax * ap.pullyR / 2]
    
    ser.reset_input_buffer()
    
    input("Remove calibration device")
    
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.reset_input_buffer()
    
    input("Turn on Power to Motors")

    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.reset_input_buffer()
    
    input("Enter to Start")
    time.sleep(2.0)
    
    gc.collect()
    
    track = None
    
    ser.reset_input_buffer()
    
    while ser.in_waiting < (num_points+1) * 11:
        pass
    
    for _ in range(11):
        get_mallet(ser)
    
    ser.reset_input_buffer()
    
    while ser.in_waiting < (num_points+1) * 11:
            pass

    xf = np.array([table_bounds[0]/4, table_bounds[1]/2])
    Vo = np.array([1, 1])
    
    get_mallet(ser)
    pos, vel, acc = get_init_conditions()
    
    #data = ap.update_path(pos, vel, acc, xf, Vo)

    #ser.write(b'\n' + data + b'\n')
   
    idx = 0
    while True:
    
        ser.reset_input_buffer()
        
        while ser.in_waiting < (num_points+1) * 11:
            pass

        if idx == 0:
            xf = np.array([0.85, 0.3])
        elif idx == 1:
            xf = np.array([0.85, 0.85])
        elif idx == 2:
            xf = np.array([0.3, 0.85])
        elif idx == 3:
            xf = np.array([0.3, 0.3])
	        
        idx += 1
        if idx == 4:
            idx = 0
	    
        Vo = np.array([3, 3])

        get_mallet(ser)
        pos, vel, acc = get_init_conditions()

        data = ap.update_path(pos, vel, acc, xf, Vo)

        ser.write(b'\n' + data + b'\n')

        time.sleep(1)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    if os.geteuid() != 0:
        print("Warning: Not running as root. Real-time performance may be limited.")
        print("Run with: sudo python3 script.py")
        return
        

    set_realtime_priority()
    
    check_isolation_status()

    system_loop()

if __name__ == "__main__":
    main()
