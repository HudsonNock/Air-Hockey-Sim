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
import cv2
import serial
import struct
import termios

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
        
def get_mallet(ser):
    if not ser.in_waiting:
        return None, None, None, False
    
    FMT = '<hhhhhhB'    # 6×int16, 1×uint8
    FRAME_SIZE = struct.calcsize(FMT)
    
    def deq(q, xmin, xmax):
        y = q/32767
        return (y+1)/2 * (xmax - xmin) + xmin
    
    # Read entire buffer
    buffer = ser.read(max(ser.in_waiting, 33))
    
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
    
    pos = np.array([deq(p0, 0, 10), deq(p1, -1, 2)])
    vel = np.array([deq(v0, -30, 30), deq(v1, -30, 30)])
    acc = np.array([deq(a0, -150, 150), deq(a1, -150, 150)])
    
    return pos, vel, acc, True

def precise_timing_measurement(N_TESTS=2000):
    """Optimized timing measurement with minimal overhead"""
    
    # Pre-allocate arrays
    times = np.zeros(N_TESTS, dtype=np.float64)
    
    # Disable garbage collection during measurement

    
    try:

        PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
        BAUD = 460800

        # === CONNECT ===
        ser = serial.Serial(PORT, BAUD, timeout=1)

        time.sleep(2)  # Wait for serial connection to settle

        # === LOOP ===

        ser.flushInput()
        for i in range(N_TESTS):
            passed = False
            time.sleep(0.005)
            start = time.perf_counter()
            while not passed:
                pos, vel, acc, passed = get_mallet(ser)
            #time.sleep(0.6)
            #print(pos[0])
            times[i] = (time.perf_counter() - start) * 1000
            #print(times[i])
            #times[i] = pos[0]

    except Exception as e:
        print("err")
        print(e)
    
    # Analyze timing
    avg_interval = np.mean(times)
    std_dev = np.std(times)
    min_interval = np.min(times)
    max_interval = np.max(times)
    
    print(f"\n=== Optimized Timing Analysis ===")
    print(f"Average interval: {avg_interval:.4f}ms")
    print(f"Standard deviation: {std_dev:.4f}ms")
    print(f"Min interval: {min_interval:.4f}ms")
    print(f"Max interval: {max_interval:.4f}ms")
    print(f"Jitter range: {max_interval - min_interval:.4f}ms")
    print(f"Actual FPS: {1000/avg_interval:.2f}")
    
    
    # Percentile analysis
    percentiles = [50, 90, 95, 99, 99.9]
    for p in percentiles:
        val = np.percentile(times, p)
        print(f"{p}th percentile: {val:.4f}ms")



def main():
    if os.geteuid() != 0:
        print("Warning: Not running as root. Real-time performance may be limited.")
        print("Run with: sudo python3 script.py")
        return

    try:
        set_realtime_priority()
        
        check_isolation_status()
        
        precise_timing_measurement()

    except Exception as ex: #PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)

if __name__ == "__main__":
    main()
