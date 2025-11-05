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
obs_dim = 38
obs = np.zeros((obs_dim,), dtype=np.float32)
obs[-6:] = np.array([ap.a1/ap.pullyR * 1e4, ap.a2/ap.pullyR * 1e1, ap.a3/ap.pullyR * 1e0, ap.b1/ap.pullyR * 1e4, ap.b2/ap.pullyR * 1e1, ap.b3/ap.pullyR * 1e1])
obs[-3] = (-6.5e-06)/ap.pullyR * 1e4

obs_flip = np.empty((obs_dim), dtype=np.float32)

margin = 0.03
margin_bottom = 0.03

mallet_r = 0.1011 / 2
puck_r = 0.0629 / 2

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


def configure_buffer_handling(cam):
    nodemap_tldevice = cam.GetTLStreamNodeMap()

    # --- Step 1: Set Buffer Count Mode to Manual ---
    try:
        buffer_size_mode = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode("StreamBufferCountMode"))
        if PySpin.IsAvailable(buffer_size_mode) and PySpin.IsWritable(buffer_size_mode):
            manual_mode = buffer_size_mode.GetEntryByName("Manual")
            if PySpin.IsAvailable(manual_mode) and PySpin.IsReadable(manual_mode):
                buffer_size_mode.SetIntValue(manual_mode.GetValue())
                print("Buffer size mode set to Manual")
    except Exception as e:
        print(f"Unable to set buffer size mode to Manual: {e}")

    # --- Step 2: Set the Buffer Count ---
    try:
        buffer_count = PySpin.CIntegerPtr(nodemap_tldevice.GetNode("StreamBufferCountManual"))
        if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
            min_val = buffer_count.GetMin()
            print(f"Minimum allowed buffer count: {min_val}")
            if min_val <= 2:
                buffer_count.SetValue(2)
                print("Buffer count set to 2")
            else:
                buffer_count.SetValue(min_val)
                print(f"Buffer count set to minimum allowed: {min_val}")
    except Exception as e:
        print(f"Unable to set buffer count: {e}")

    # --- Step 3: Set Stream Buffer Handling Mode ---
    buffer_handling_mode = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode("StreamBufferHandlingMode"))
    if PySpin.IsAvailable(buffer_handling_mode) and PySpin.IsWritable(buffer_handling_mode):
        newest_only = buffer_handling_mode.GetEntryByName("NewestOnly")
        if PySpin.IsAvailable(newest_only) and PySpin.IsReadable(newest_only):
            buffer_handling_mode.SetIntValue(newest_only.GetValue())
            print("Buffer Handling Mode set to NewestOnly")
        else:
            print("NewestOnly option not available")
    else:
        print("Unable to set Buffer Handling Mode")

def configure_camera(cam, gain_val=30.0, exposure_val=100.0):
    # Get the camera node map
    nodemap = cam.GetNodeMap()
    
    auto_modes = ["ExposureAuto", "GainAuto", "BalanceWhiteAuto", "GammaEnable"]
    for mode in auto_modes:
        try:
            auto_node = PySpin.CEnumerationPtr(nodemap.GetNode(mode))
            if PySpin.IsAvailable(auto_node) and PySpin.IsWritable(auto_node):
                off_entry = auto_node.GetEntryByName("Off")
                if PySpin.IsAvailable(off_entry):
                    auto_node.SetIntValue(off_entry.GetValue())
                    print(f"{mode} disabled")
        except:
            print("Unable to disable " + mode)

    exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if PySpin.IsAvailable(exposure_time) and PySpin.IsWritable(exposure_time):
        # ExposureTime is in microseconds
        exposure_time.SetValue(exposure_val)
    else:
        print("Unable to set ExposureTime.")

    gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsAvailable(gain) and PySpin.IsWritable(gain):
        gain.SetValue(gain_val)
    else:
        print("Unable to set Gain.")
        
    # Try to disable image processing features that add latency
    #processing_features = ["GammaEnable"] #, "SharpnessEnable", "SaturationEnable", 
                          #"HueEnable", "DefectCorrectStaticEnable"]
    
    #for feature in processing_features:
    #    try:
    #        feature_node = PySpin.CBooleanPtr(nodemap.GetNode(feature))
    #        if PySpin.IsAvailable(feature_node) and PySpin.IsWritable(feature_node):
    #            feature_node.SetValue(False)
    #            print(f"{feature} disabled")
    #    except:
    #        print("Unable to disable " + feature)
    
    try:
        balance_ratio = PySpin.CEnumerationPtr(nodemap.GetNode("BalanceRatioSelector"))
        if PySpin.IsAvailable(balance_ratio) and PySpin.IsWritable(balance_ratio):
            blue_entry = balance_ratio.GetEntryByName("Blue")
            if PySpin.IsAvailable(blue_entry):
                balance_ratio.SetIntValue(blue_entry.GetValue())
                print("Balance ratio Selector set to Blue")
    except:
        print("Unable to set balance ratio selector")
        
    try:
        balance_ratio = PySpin.CFloatPtr(nodemap.GetNode("BalanceRatio"))
        if PySpin.IsAvailable(balance_ratio) and PySpin.IsWritable(balance_ratio):
            balance_ratio.SetValue(3.6)
            print("balance ratio set to 3.6")
    except:
        print("Unable to set balance ratio")
            
    throughput_limit = PySpin.CIntegerPtr(nodemap.GetNode("DeviceLinkThroughputLimit"))
    if PySpin.IsAvailable(throughput_limit) and PySpin.IsWritable(throughput_limit):
        max_value = throughput_limit.GetMax()
        throughput_limit.SetValue(max_value)
        print(f"Set throughput limit to {max_value}")
    else:
        print("Unable to read or write throughput limit mode.")


def set_pixel_format(cam, mode="BayerRG8"):
    nodemap = cam.GetNodeMap()
    
    pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
        # Try Mono8 first (fastest)
        format_entry = pixel_format.GetEntryByName(mode)
        if PySpin.IsAvailable(format_entry) and PySpin.IsReadable(format_entry):
            pixel_format.SetIntValue(format_entry.GetValue())
            print("Pixel format set to " + mode)
            
        else:
            print("unable to set pixel format to " + mode)
    else:
        print("Unable to read pixel format")
        
    if mode=="BayerRG8":
        ISP = PySpin.CBooleanPtr(nodemap.GetNode("IspEnable"))
        if PySpin.IsAvailable(ISP) and PySpin.IsWritable(ISP):
            ISP.SetValue(False)
        else:
            print("Unable to turn off ISP")

        return False

def configure_gain(cam, gain_val = 30.0):
    nodemap = cam.GetNodeMap()
    gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsAvailable(gain) and PySpin.IsWritable(gain):
        gain.SetValue(gain_val)
    else:
        print("Unable to set Gain.")

def set_frame_rate(cam, target_fps=120.0):
    nodemap = cam.GetNodeMap()
    
    # Enable frame rate control
    frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsAvailable(frame_rate_enable) and PySpin.IsWritable(frame_rate_enable):
        frame_rate_enable.SetValue(True)
        print("Frame rate control enabled")
    else:
        print("Unable to enable frame rate control")
        return 0

    # Set specific frame rate
    frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if PySpin.IsAvailable(frame_rate) and PySpin.IsWritable(frame_rate):
        min_fps = frame_rate.GetMin()
        max_fps = frame_rate.GetMax()
        print(f"Frame rate range: {min_fps:.2f} - {max_fps:.2f} fps")
        
        # Set to target or maximum available
        actual_fps = min(target_fps, max_fps)
        frame_rate.SetValue(actual_fps)
        print(f"Frame rate set to: {actual_fps:.2f} fps")
        return actual_fps
    else:
        print("Unable to set frame rate")
        return 0
    
def set_roi(cam, width, height, offset_x=0, offset_y=0):
    nodemap = cam.GetNodeMap()
    
    # Set width
    width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
    if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
        width_node.SetValue(width)
    
    # Set height  
    height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
    if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
        height_node.SetValue(height)
        
    # Set offsets
    offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
    if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
        offset_x_node.SetValue(offset_x)
        
    offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
    if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
        offset_y_node.SetValue(offset_y)
        
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
    
    pos = np.array([coef_x[1]*t + coef_x[2], \
                    coef_y[1]*t + coef_y[2]])
                    #np.array([coef_x[0]*t**2 + coef_x[1]*t + coef_x[2], \
                    #coef_y[0]*t**2 + coef_y[1]*t + coef_y[2]])
                    
    vel = np.array([2*coef_x[0]*t + coef_x[1], \
                    2*coef_y[0]*t + coef_y[1]])
                    
    acc = np.array([2*coef_x[0], \
                    2*coef_y[0]])
                    
    return pos, vel, acc
    
def apply_symmetry(t_obs):
    obs_flip[:] = t_obs
    obs_flip[1:22:2] = table_bounds[1] - obs_flip[1:22:2]
    obs_flip[23] *= -1
    obs_flip[27] *= -1
    obs_flip[25] = table_bounds[1] - obs_flip[25]
    obs_flip[29] = table_bounds[1] - obs_flip[29]

    return obs_flip
    

def system_loop(cam, load, pro):
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement
    img_shape = (1536, 1296)
    opponent_mallet_z = 68.35*10**(-3)
    if not pro:
        opponent_mallet_z = 120.94*10**(-3)

    PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
    BAUD = 460800

    # === CONNECT ===
    ser = serial.Serial(PORT, BAUD, timeout=0)
    if not load:
        time.sleep(1)
        
        set_pixel_format(cam, mode="Mono8")
        configure_camera(cam, gain_val=1.0, exposure_val=10000.0)
        set_frame_rate(cam, target_fps=20.0)
        
        cam.BeginAcquisition()

        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
            
        bright_filter = -0.5 * np.arange(1536)[:,None] / 1536 + 1.5

        image = cam.GetNextImage()
        img = np.clip(image.GetData().reshape(img_shape) * bright_filter, 0, 255).astype(np.uint8)
        image.Release()
        
        #while True:
        #    cv2.imshow("arucos", img)
        #    cv2.waitKey(1)
        #    image = cam.GetNextImage()
        #    img = image.GetData().reshape(img_shape)
        #    image.Release()
        
        setup = tracker.SetupCamera()
        while not setup.run_extrinsics(img):
            cv2.imshow("arucos", img[::2, ::2])
            cv2.waitKey(0)
            image = cam.GetNextImage()
            img = np.clip(image.GetData().reshape(img_shape) * bright_filter, 0, 255).astype(np.uint8)
            image.Release()
            
        cv2.destroyAllWindows()
        
        cam.EndAcquisition()
        
    set_pixel_format(cam, mode="BayerRG8")
    configure_camera(cam, gain_val=33.0, exposure_val=100.0)
    set_frame_rate(cam, target_fps=120.0)
    
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
    
    a1 = 7.474*10**(-6) #3.579*10**(-6)
    a2 = 7.575e-03 #6.721*10**(-3) 
    a3 = 6.969e-02
    b1 = -1.607*10**(-6) #-1.7165*10**(-6)
    b2 = -2.838e-03 #-2.731*10**(-3)
    b3 = 3.688e-03
    
    obs[-6:] = np.array([a1/ap.pullyR * 1e4, a2/ap.pullyR * 1e1, a3/ap.pullyR * 1e0, (-6.5e-06)/ap.pullyR * 1e4, b2/ap.pullyR * 1e1, b3/ap.pullyR * 1e1]) * 1.1
    
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
    
    cam.BeginAcquisition()
    track = None
    
    ser.reset_input_buffer()
    
    while ser.in_waiting < (num_points+1) * 11:
        pass
    
    for _ in range(11):
        get_mallet(ser)
    
    if not load:
        xf = np.array([0.7, 0.2])
        Vo = np.array([5, 5])
        
        get_mallet(ser)
        pos, vel, acc = get_init_conditions()

        data = ap.update_path(pos, vel, acc, xf, Vo)

        ser.write(b'\n' + data + b'\n')

        time.sleep(1)
        
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
        
        y_max = np.max(img[:, int(img.shape[1]/2-30):int(img.shape[1]/2+30)], axis=1)
        cutoff = np.array([np.max(y_max[max(0,i-30):min(i+30, len(y_max))]) for i in range(len(y_max))])
        
        xf = np.array([0.2, 0.2])
        Vo = np.array([5, 5])
        
        ser.reset_input_buffer()
        
        while ser.in_waiting < (num_points+1) * 11:
            pass
        
        for _ in range(11):
            get_mallet(ser)
        
        pos, vel, acc = get_init_conditions()

        data = ap.update_path(pos, vel, acc, xf, Vo)

        ser.write(b'\n' + data + b'\n')
        
        time.sleep(1)
        
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
        
        y_max = np.max(img[:, int(img.shape[1]/2-30):int(img.shape[1]/2+30)], axis=1)
        cutoff = np.maximum(cutoff, np.array([np.max(y_max[max(0,i-30):min(i+30, len(y_max))]) for i in range(len(y_max))]))
        
        del y_max
    
        track = tracker.CameraTracker(setup.rotation_matrix,
                                      setup.translation_vector,
                                      setup.z_pixel_map,
                                      68.35*10**(-3),
                                      cutoff)
                                  
        np.savez("setup_data.npz",
                rotation_matrix=setup.rotation_matrix,
                translation_vector=setup.translation_vector,
                z_pixel_map=setup.z_pixel_map,
                cutoff=cutoff)
                                  
        del setup
    else:
        setup_data = np.load("setup_data.npz")
        track = tracker.CameraTracker(setup_data["rotation_matrix"],
                                      setup_data["translation_vector"],
                                      setup_data["z_pixel_map"],
                                      opponent_mallet_z,
                                      setup_data["cutoff"])

    gc.collect()
    
    for _ in range(20):
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        track.process_frame(img)
        image.Release()
        
    dt = 4.0768/1000
    ser.reset_input_buffer()
    
    while ser.in_waiting < (num_points+1) * 11:
        pass
        
    for _ in range(11):
        get_mallet(ser)
    #timer = time.perf_counter()
    
    recording_data = np.zeros([20000, 7])
    idx = 0
    timer = time.perf_counter()
    left_hysteresis = False
    symmetry = False
    while True:
    
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        track.process_frame(img)
        image.Release()
        
        get_mallet(ser)
        pos, vel, acc = get_init_conditions()
            
        new_time = time.perf_counter()
        time_diff = new_time - timer
        timer = new_time
        obs[:20] = track.past_data.get()
        obs[24:26] = obs[20:22] #update past mallet
        obs[26:28] = obs[22:24]
        obs[20:22] = pos #add current mallet pos
        obs[22:24] = vel
        
        if left_hysteresis and obs[0] > table_bounds[0]/2 + 0.1:
            left_hysteresis = False
            symmetry = np.random.random() < 0.5
        elif (not left_hysteresis) and obs[0] < table_bounds[0]/2 - 0.1:
            left_hysteresis = True
        
        recording_data[idx,:2] = obs[:2]
        recording_data[idx,2:6] = obs[20:24]
        recording_data[idx,6] = time_diff
        idx += 1
        
        if idx == len(recording_data):
            with open("system_loop_data_N6.csv", "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["Px", "Py", "Mx", "My", "Mxv", "Myv", "dt"])
                
                # Write rows
                for i in range(len(recording_data)):
                    writer.writerow([recording_data[i, 0], recording_data[i, 1], recording_data[i, 2], recording_data[i, 3], recording_data[i, 4], recording_data[i,5], recording_data[i,6]])
            print("SIGNAL END")
            break
        
        
        #print("--")
        #print(obs)
                       
        
        if symmetry:
            tensor_obs = TensorDict({"observation": torch.tensor(obs, dtype=torch.float32)})
            with torch.no_grad():
                policy_out = ap.policy_module(tensor_obs)
            action = policy_out["action"].detach().numpy()
        else:
            tensor_obs = TensorDict({"observation": torch.tensor(apply_symmetry(obs), dtype=torch.float32)})
            with torch.no_grad():
                policy_out = ap.policy_module(tensor_obs)
            action = policy_out["action"].detach().numpy()
            action[1] = table_bounds[1] - action[1]

        
        obs[28:32] = action
        #print(action)
        no_update_bounds = 0.08
        no_update = (np.linalg.norm(obs[20:22] - obs[24:26]) < no_update_bounds) and (np.linalg.norm(obs[28:30] - action[:2]) < no_update_bounds) and (np.linalg.norm(action[:2] - obs[20:22]) < no_update_bounds)
        
        if not no_update:
            #print('--')
            #print(obs)
            #print(action)
            xf = action[:2]

            xf[0] = np.maximum(margin_bottom+mallet_r, xf[0])
            xf[0] = np.minimum(table_bounds[0]/2-mallet_r, xf[0])

            xf[1] = np.maximum(margin+mallet_r, xf[1])
            xf[1] = np.minimum(table_bounds[1]-margin-mallet_r, xf[1])
            
            #if time.perf_counter() - timer > 4:
            #    top = not top
            #    timer = time.perf_counter()
            
            #if top:
            #    xf[0] = table_bounds[0]/2 - 0.25
            #    xf[1] = table_bounds[1]/2
            #else:
            #    xf[0] = 0.25
            #    xf[1] = table_bounds[1]/2
            

            Vo = action[2] * Vmax * np.array([1+action[3],1-action[3]])
            #Vo[0] = np.minimum(Vo[0], 3)
            #Vo[1] = np.minimum(Vo[1], 3)
            #print("A")
            #print(xf)
            #print(Vo)
            
            #Vo[0] = 7
            #Vo[1] = 7
            #obs[28:32] = np.concatenate([xf, Vo], axis=0)
            get_mallet(ser)
            mallet_data = mallet_buffer.read()
            pos, vel, acc = get_init_conditions(pred = dt)

            #new_pos = pos + vel * dt + 0.5 * acc * dt**2
            #new_vel = vel + acc * dt
            data = ap.update_path(pos, vel, acc, xf, Vo)
            ser.write(b'\n' + data + b'\n')
        
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        track.process_frame(img)
        image.Release()
        
        get_mallet(ser)
        pos, vel, acc = get_init_conditions()
            
        new_time = time.perf_counter()
        time_diff = new_time - timer
        timer = new_time
        
        recording_data[idx,:2] = track.past_data.get()[:2]
        recording_data[idx,2:4] = pos
        recording_data[idx,4:6] = vel
        recording_data[idx,6] = time_diff
        idx += 1
        
        if idx == len(recording_data):
            with open("system_loop_data_N6.csv", "w", newline="") as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["Px", "Py", "Mx", "My", "Mxv", "Myv", "dt"])
                
                # Write rows
                for i in range(len(recording_data)):
                    writer.writerow([recording_data[i, 0], recording_data[i, 1], recording_data[i, 2], recording_data[i, 3], recording_data[i, 4], recording_data[i,5], recording_data[i,6]])
            print("SIGNAL END")
            break
    
    cam.EndAcquisition()

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
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str2bool, default=False)
    parser.add_argument("--pro", type=str2bool, default=True)
    args = parser.parse_args()
    
    system = PySpin.System.GetInstance()
    
    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    
    if num_cameras == 0:
        print("No cameras detected!")
        cam_list.Clear()
        system.ReleaseInstance()
        return
    
    # Select the first camera
    cam = cam_list.GetByIndex(0)
    cam.Init()
    set_realtime_priority()
    
    check_isolation_status()
    
    configure_buffer_handling(cam)
    set_roi(cam,1296,1536,376,0)

    system_loop(cam, args.load, args.pro)

if __name__ == "__main__":
    main()
