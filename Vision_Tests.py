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

torch.set_num_threads(2)
torch.set_num_interop_threads(1)


table_bounds = np.array([1.9885, 0.995])

margin = 0.065
margin_bottom = 0.1

mallet_r = 0.0508
margin_bounds = 0.0
mallet_bounds = np.array([[margin_bounds + mallet_r, table_bounds[0]/2  + mallet_r/2], [margin_bounds+mallet_r, table_bounds[1]-margin_bounds-mallet_r]])

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
            balance_ratio.SetValue(3.3)
            print("balance ratio set to 3.3")
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
    
    pos = np.array([deq(p0, 0, 30), deq(p1, -1, 2)])
    vel = np.array([deq(v0, -30, 30), deq(v1, -30, 30)])
    acc = np.array([deq(a0, -150, 150), deq(a1, -150, 150)])
    
    return pos, vel, acc, True

def object_loc(cam):
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement

    try:
        set_pixel_format(cam, mode="Mono8")
        configure_camera(cam, gain_val=10.0, exposure_val=6000.0)
        set_frame_rate(cam, target_fps=20.0)
        
        cam.BeginAcquisition()
        
        # Warm up - discard first few frames
        img_shape = (1536, 1296)

        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
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
            img = image.GetData().reshape(img_shape)
            image.Release()
            
        cv2.destroyAllWindows()
        
        cam.EndAcquisition()
        
        track = tracker.CameraTracker(setup.rotation_matrix,
                                      setup.translation_vector,
                                      setup.z_pixel_map,
                                      68.35*10**(-3)) #70.44*10**(-3)) #(120.94)*10**(-3))
                                      
        set_pixel_format(cam, mode="BayerRG8")
        configure_camera(cam, gain_val=35.0, exposure_val=100.0)
        set_frame_rate(cam, target_fps=120.0)
        
        del setup
        
        gc.collect()
        
        cam.BeginAcquisition()
        
        while True:
            
            image = cam.GetNextImage()
            #print("P")
            img = image.GetData().reshape(img_shape)
            #print("U")
            image.Release()
            track.process_frame(img, top_down_view=True, printing=False)
            #print("Q")
        
        cam.EndAcquisition()
    except Exception as e:
        print("err")
        print(e)


def main():
    if os.geteuid() != 0:
        print("Warning: Not running as root. Real-time performance may be limited.")
        print("Run with: sudo python3 script.py")
        return
    
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
    try:
        cam.Init()
        set_realtime_priority()
        
        check_isolation_status()
        
        configure_buffer_handling(cam)
        set_roi(cam,1296,1536,376,0)

        object_loc(cam)

    except Exception as ex: #PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
    finally:
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()

if __name__ == "__main__":
    main()
