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
import csv

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

table_bounds = np.array([2.362, 1.144])

margin = 0.05
margin_bottom = 0.05

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

    FMT = '<hhhhhhhB'    # 5×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)

    buffer = bytearray()
    buffer.extend(ser.read(ser.in_waiting))
    print(buffer)
            
    # Find the start marker (0xAA)
    start_idx = 0
    if buffer[0] != 0xFF or buffer[1] != 0xFF or buffer[2] != 0xFF:
        print("ERROR")
        print(buffer)
        print(1/0)
    
    # Check if we have enough bytes for a complete frame after the start marker
    remaining_bytes = len(buffer) - start_idx - 1  # -1 for the start marker itself
    if remaining_bytes < FRAME_SIZE + 1:  # +1 for end marker
        print(1/0)
    
    # Extract the frame data
    frame_start = 3
    frame_end = frame_start + FRAME_SIZE
    raw = buffer[frame_start:frame_end]
    
    if frame_end >= len(buffer) or buffer[frame_end] != 0x55:
        print("Invalid end marker")
        print(1/0)
    
    buffer = buffer[frame_end+1:]

    # Unpack the data
    p0, p1, ep0, ep1, pwm0, pwm1, dt, chk = struct.unpack(FMT, raw)

    # Verify checksum
    c = 0
    for b in raw[:-1]:
        c ^= b
    if c != chk:
        print("bad checksum")
        print(1/0)

    pos = np.array([deq(p0, -0.5, 2), deq(p1, -0.5, 2)])
    
    return pos
        
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

def collect_data():
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement
    #try:
    
    action_commands = []#np.load('data/actions_newp.npy')
    """
    for idx in range(6):
        if idx%4 == 0:
            xf = np.array([0.85, 0.3])
        elif idx%4 == 1:
            xf = np.array([0.85, 0.85])
        elif idx%4 == 2:
            xf = np.array([0.3, 0.85])
        elif idx%4 == 3:
            xf = np.array([0.3, 0.3])
        action_commands.append(np.concatenate((xf, np.array([5.0,5.0]), np.array([1.0])), axis=0))
    action_commands = np.array(action_commands)
    """
    for idx in range(50):
        xf = np.array([0.3+np.random.random()*(0.85-0.3), 0.3+np.random.random()*(0.85-0.3)])
        action_commands.append(np.concatenate((xf, np.array([(24*0.8-3)*np.random.random() + 3.0, (24*0.8-3)*np.random.random() + 3.0]), np.array([(0.2-0.13)*np.random.random() + 0.13])), axis=0))
        
    #np.array([3, 3]), np.array([0.2])), axis=0))
    action_commands = np.array(action_commands)

    
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
    
    time.sleep(1.0)
    pos = get_mallet(ser)
            
    xf = np.array([table_bounds[0]/4, table_bounds[1]/2])
    Vo = np.array([3, 3])
    
    data = ap.update_path(pos, np.zeros((2,)), np.zeros((2,)), xf, Vo)
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
    pos[0,:] = pully_R
    idx = 1
    
    #delay = 0.02 #np.random.random() * 0.1 + 0.02 #0.3 + 0.2
    action_idx = 0
    ser.reset_input_buffer()
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
            
            if frame_end >= len(buffer) or buffer[frame_end] != 0x55:
                print("Invalid end marker")
                print(1/0)
                break
    
            buffer = buffer[frame_end+1:]

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
            with open("new_data/mallet_data_random_supercap_feedback_10_MaxV.csv", "w", newline="") as f:
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
