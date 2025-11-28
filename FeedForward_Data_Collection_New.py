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
import agent_processing as ap
import argparse
import csv

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

table_bounds = np.array([1.993, 0.992])

margin = 0.03
margin_bottom = 0.03

mallet_r = 0.1011 / 2
puck_r = 0.0629 / 2

Vmax = 24 * 0.8
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
    y = q/32767.0
    return (y+1)/2 * (xmax - xmin) + xmin
    
def get_mallet(ser):    
    FMT = '<hhhhhhhB'    # 6×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)
    
    # Read entire buffer
    #while ser.in_waiting < 33:
    #    pass
    search_buffer = ser.read(ser.in_waiting)
    
    # Start from the last 29 bytes
    
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
        print("Invalid end marker")
        return None, None, None, False
    
    # Unpack the data
    p0, p1, ep0, ep1, pwm0, pwm1, dt, chk = struct.unpack(FMT, raw)

    # Verify checksum
    c = 0
    for b in raw[:-1]:
        c ^= b
    if c != chk:
        print("bad checksum", c, chk)
        return None, None, None, False
    
    pos = np.array([deq(p0, -0.5, 2), deq(p1, -0.5, 2)])
    vel = np.array([0,0])
    acc = np.array([0,0])
    
    return pos, vel, acc, True

def collect_data():
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement
    #try:
    
    action_commands = np.load('data/actions_newp.npy')

    
    PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
    BAUD = 460800

    ser = serial.Serial(PORT, BAUD, timeout=0)
    
    ser.write(b'\n')
    while ser.in_waiting == 0:
        continue
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
    gc.collect()
    
    time.sleep(0.5)
    passed = False
    while not passed:
        pos, vel, acc, passed = get_mallet(ser)
            
    xf = np.array([0.7, 0.5])
    Vo = np.array([5, 5]) #np.array([15, 13])
    
    data = ap.update_path(pos, vel, acc, xf, Vo)
    ser.write(b'\n' + data + b'\n')
    
    buffer = bytearray()
    
    FMT = '<hhhhhhhB'    # 5×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)
    
    sample_len = 70000
    pos = np.zeros((sample_len,2))
    exp_pos = np.zeros((sample_len,2))
    pwms = np.zeros((sample_len,2))
    dts = np.zeros((sample_len,))
    
    signal_end = False
    
    t1 = time.perf_counter()
    counter = 0
    pos[0,:] = pully_R
    idx = 1
    
    #delay = 0.02 #np.random.random() * 0.1 + 0.02 #0.3 + 0.2
    action_idx = 0
    while True:
        # Read entire buffer
        
        if time.perf_counter() - t1 > action_commands[action_idx, 4]:
            time_passed = time.perf_counter() - t1
            t1 = time.perf_counter()
            
            #xf = np.array([np.random.random() * (0.4) + 0.3, np.random.random() * 0.4 + 0.3])
            #Vo = np.array([np.random.random() * (24*0.8-10) + 10, np.random.random() * (24*0.8-10) + 10])
            xf = action_commands[action_idx, :2]
            xf[0] = max(xf[0], 0.3+(action_idx%2)*0.01)
            xf[0] = min(xf[0], 0.7+(action_idx%2)*0.01)
            xf[1] = max(xf[1], 0.3+(action_idx%2)*0.01)
            xf[1] = min(xf[1], 0.7+(action_idx%2)*0.01)
            
            Vo = action_commands[action_idx, 2:4]
            action_idx += 1
            if action_idx == len(action_commands):
                print("END OF ACTIONS")
                signal_end = True
            #Vo = np.array([12,12])
            #delay = 0.02 #np.random.random() * 0.1 + 0.02 #0.3 + 0.2
            
            pos_ic, vel_ic, acc_ic = ap.get_IC(time_passed)

            data = ap.update_path(pos_ic, vel_ic, acc_ic, xf, Vo)
            ser.write(b'\n' + data + b'\n')
            
        
           
        if ser.in_waiting > 1000:
            print(ser.in_waiting)
            print(1/0)
        while ser.in_waiting < 33:
            pass
        buffer.extend(ser.read(ser.in_waiting))
            
        while len(buffer) > 60 and not signal_end:
            # Find the start marker (0xAA)
            start_idx = 0
            if buffer[0] != 0xFF or buffer[1] != 0xFF or buffer[2] != 0xFF:
                print("ERROR")
                print(buffer)
                print(1/0)
                break
            
            # Check if we have enough bytes for a complete frame after the start marker
            remaining_bytes = len(buffer) - start_idx - 1  # -1 for the start marker itself
            if remaining_bytes < FRAME_SIZE + 1:  # +1 for end marker
                break
            
            # Extract the frame data
            frame_start = 3
            frame_end = frame_start + FRAME_SIZE
            raw = buffer[frame_start:frame_end]
            buffer = buffer[frame_end+1:]

            # Check end marker
            if frame_end >= len(buffer) or buffer[frame_end] != 0x55:
                print("Invalid end marker")
                break

            # Unpack the data
            p0, p1, ep0, ep1, pwm0, pwm1, dt, chk = struct.unpack(FMT, raw)

            # Verify checksum
            c = 0
            for b in raw[:-1]:
                c ^= b
            if c != chk:
                print("bad checksum")

            pos[idx, :] = np.array([deq(p0, -0.5, 2), deq(p1, -0.5, 2)])
            exp_pos[idx, :] = np.array([deq(ep0, -0.5, 2), deq(ep1, -0.5, 2)])
            pwms[idx,:] = np.array([deq(pwm0, -1.1, 1.1), deq(pwm1, -1.1, 1.1)])
            dts[idx] = deq(dt, 0, 3)
    
            #if abs(deq(v0, -1.1, 1.1)) < 0.02 and abs(deq(v1, -1.1, 1.1)) < 0.02:
            #    #print(abs(deq(v0, -1.1, 1.1)))
            #    #print(abs(deq(v1, -1.1, 1.1)))
            #    #print("B")
            #    signal_end = True
            #    break
            
            idx += 1
            
            if idx == sample_len:
                signal_end = True
                break
            
                
        if signal_end:
            #print("------")
            #print(pos)
            #print("------")
            #print(pwms)
            #print("------")
            #print(dts)
            with open("data/mallet_data_newp_supercap4.csv", "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["x", "y", "Expected_x", "Expected_y", "Left_PWM", "Right_PWM", "dt"])
                
                # Write rows
                for p, ep, pwm, dt in zip(pos, exp_pos, pwms, dts):
                    writer.writerow([p[0], p[1], ep[0], ep[1], pwm[0], pwm[1], dt])
            print("SIGNAL END")
            break
                
            
    #except Exception as e:
    #    print("err")
    #    print(e)

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

    collect_data()

if __name__ == "__main__":
    main()
