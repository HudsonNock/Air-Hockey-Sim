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
import argparse

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

table_bounds = np.array([2.362, 1.144])

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


def configure_camera(cam, gain_val=30.0, exposure_val=100.0, gamma_val=None, black_level_val=0, balance_ratio_val=3.6):
    # Get the camera node map
    nodemap = cam.GetNodeMap()
    
    auto_modes = ["ExposureAuto", "GainAuto", "BalanceWhiteAuto"]
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
            
    if gamma_val is None:
        try:
            auto_node = PySpin.CEnumerationPtr(nodemap.GetNode("GammaEnable"))
            if PySpin.IsAvailable(auto_node) and PySpin.IsWritable(auto_node):
                off_entry = auto_node.GetEntryByName("Off")
                if PySpin.IsAvailable(off_entry):
                    auto_node.SetIntValue(off_entry.GetValue())
                    print("GammaEnable disabled")
        except:
            print("Unable to disable GammaEnable")
    else:
        try:
            auto_node = PySpin.CEnumerationPtr(nodemap.GetNode("GammaEnable"))
            if PySpin.IsAvailable(auto_node) and PySpin.IsWritable(auto_node):
                on_entry = auto_node.GetEntryByName("On")
                if PySpin.IsAvailable(on_entry):
                    auto_node.SetIntValue(on_entry.GetValue())
                    print("GammaEnable enabled")
                    
                    gamma = PySpin.CFloatPtr(nodemap.GetNode("Gamma"))
                    if PySpin.IsAvailable(gamma) and PySpin.IsWritable(gamma):
                        gamma.SetValue(gamma_val)
                        print(f"Set Gamma to {gamma_val}")
        except:
            print("Unable to set Gamma")

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
            balance_ratio.SetValue(balance_ratio_val)
            print(f"balance ratio set to {balance_ratio_val}")
    except:
        print("Unable to set balance ratio")
            
    throughput_limit = PySpin.CIntegerPtr(nodemap.GetNode("DeviceLinkThroughputLimit"))
    if PySpin.IsAvailable(throughput_limit) and PySpin.IsWritable(throughput_limit):
        max_value = throughput_limit.GetMax()
        throughput_limit.SetValue(max_value)
        print(f"Set throughput limit to {max_value}")
    else:
        print("Unable to read or write throughput limit mode.")
        
                
    try:
        auto_node = PySpin.CEnumerationPtr(nodemap.GetNode("BlackLevelSelector"))
        if PySpin.IsAvailable(auto_node) and PySpin.IsWritable(auto_node):
            all_entry = auto_node.GetEntryByName("All")
            if PySpin.IsAvailable(all_entry):
                auto_node.SetIntValue(all_entry.GetValue())
                print("Black Level set to all")
    except:
        print("Unable to set black level to all")

    # 1. Turn off Auto Black Level first (otherwise BlackLevel is often read-only)
    auto_mode_node = PySpin.CEnumerationPtr(nodemap.GetNode("BlackLevelAuto"))
    if PySpin.IsAvailable(auto_mode_node) and PySpin.IsWritable(auto_mode_node):
        off_entry = auto_mode_node.GetEntryByName("Off")
        if PySpin.IsAvailable(off_entry) and PySpin.IsReadable(off_entry):
            auto_mode_node.SetIntValue(off_entry.GetValue())
            print("Automatic Black Level turned Off.")

    # 2. Access the BlackLevel value node
    # Note: Usually this is a Float, not an Enumeration
    black_level_node = PySpin.CFloatPtr(nodemap.GetNode("BlackLevel"))

    if PySpin.IsAvailable(black_level_node) and PySpin.IsWritable(black_level_node):
        # Ensure the value is within the camera's allowed range
        val_to_set = max(black_level_node.GetMin(), min(black_level_node.GetMax(), black_level_val))
        black_level_node.SetValue(val_to_set)
        print(f"Set Black level to {val_to_set}")
    else:
        print("BlackLevel node is still not writable. Check 'BlackLevelSelector'.")


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
    try:
        width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            width_node.SetValue(width)
        
        # Set height  
        height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            height_node.SetValue(height)
            
        offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            offset_x_node.SetValue(offset_x)
            
        offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            offset_y_node.SetValue(offset_y)
    except Exception as e:
        offset_x_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetX"))
        if PySpin.IsAvailable(offset_x_node) and PySpin.IsWritable(offset_x_node):
            offset_x_node.SetValue(offset_x)
            
        offset_y_node = PySpin.CIntegerPtr(nodemap.GetNode("OffsetY"))
        if PySpin.IsAvailable(offset_y_node) and PySpin.IsWritable(offset_y_node):
            offset_y_node.SetValue(offset_y)
            
        width_node = PySpin.CIntegerPtr(nodemap.GetNode("Width"))
        if PySpin.IsAvailable(width_node) and PySpin.IsWritable(width_node):
            width_node.SetValue(width)
        
        # Set height  
        height_node = PySpin.CIntegerPtr(nodemap.GetNode("Height"))
        if PySpin.IsAvailable(height_node) and PySpin.IsWritable(height_node):
            height_node.SetValue(height)
    
def object_loc(cam, load):
    """Optimized timing measurement with minimal overhead"""
    
    # Disable garbage collection during measurement

    #try:
    img_shape = (1450, 1300)
    offset = (20, 396)
    if not load:
        max_img_shape = (1536, 2048)
        set_roi(cam,max_img_shape[1],max_img_shape[0],0,0)
        set_pixel_format(cam, mode="Mono8")
        configure_camera(cam, gain_val=4.0, exposure_val=8000.0, balance_ratio_val=2.5)
        set_frame_rate(cam, target_fps=10.0)
        
        cam.BeginAcquisition()
        
        bright_filter = -0.5 * np.arange(max_img_shape[0])[:,None] / 1536 + 1.5

        image = cam.GetNextImage()
        img = np.clip(image.GetData().reshape(max_img_shape) * bright_filter, 0, 255).astype(np.uint8)
        image.Release()
        
        setup = tracker.SetupCamera(img_shape=img_shape, offset=offset)
        while not setup.run_extrinsics(img):
            cv2.imshow("arucos", img[::2, ::2])
            cv2.waitKey(0)
            image = cam.GetNextImage()
            img = image.GetData().reshape(max_img_shape)
            image.Release()
        
        cv2.destroyAllWindows()
        
        cam.EndAcquisition()
        
        set_roi(cam,img_shape[1],img_shape[0],offset[1],offset[0])                     
        set_pixel_format(cam, mode="BayerRG8")
        configure_camera(cam, gain_val=25.0, exposure_val=100.0, gamma_val=1.0, black_level_val=-5, balance_ratio_val=3.4)
        set_frame_rate(cam, target_fps=120.0)
        
        print("Remove mallet + puck from view")
        input()
        
        cam.BeginAcquisition()
        
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
        
        #1105, 1259
        #cv2.imshow("vision", img[::2, ::2])
        #cv2.waitKey(0)
        
        y_max = np.max(img, axis=1)
        y_max[1259] = np.max(np.concatenate([img[:1106, 1259], np.array([0]), img[1107:, 1259]]))
        cutoff = np.array([np.max(y_max[max(0,i-30):min(i+30, len(y_max))]) for i in range(len(y_max))])
        
        for _ in range(10):
            image = cam.GetNextImage()
            img = image.GetData().reshape(img_shape)
            image.Release()
        
        print("Move mallet up and down slowly all the way (8s)")
        
        for _ in range(120*8):
            image = cam.GetNextImage()
            img = image.GetData().reshape(img_shape)
            image.Release()

            y_max = np.max(img, axis=1)
            y_max[1259] = np.max(np.concatenate([img[:1106, 1259], np.array([0]), img[1107:, 1259]]))
            cutoff2 = np.array([np.max(y_max[max(0,i-30):min(i+30, len(y_max))]) for i in range(len(y_max))])
            
            cutoff = np.maximum(cutoff, cutoff2)
        
        del y_max
        del cutoff2
        setup.set_thresh_map(cutoff)
        
        cv2.imshow("thresh", setup.thresh_map[::2, ::2])
        cv2.waitKey(0)
                
        track = tracker.CameraTracker(setup.rotation_matrix,
                                      setup.translation_vector,
                                      setup.z_pixel_map,
                                      setup.thresh_map,
                                      img_shape,
                                      offset)
                                      
        np.savez("setup_data.npz",
            rotation_matrix=setup.rotation_matrix,
            translation_vector=setup.translation_vector,
            z_pixel_map=setup.z_pixel_map,
            thresh_map=setup.thresh_map)
                                      
        del setup
        del cutoff
    else:
        setup_data = np.load("setup_data.npz")
        track = tracker.CameraTracker(setup_data["rotation_matrix"],
                                  setup_data["translation_vector"],
                                  setup_data["z_pixel_map"],
                                  setup_data["thresh_map"],
                                  img_shape,
                                  offset)
                                  
        set_roi(cam,img_shape[1],img_shape[0],offset[1],offset[0])                     
        set_pixel_format(cam, mode="BayerRG8")
        configure_camera(cam, gain_val=25.0, exposure_val=100.0, gamma_val=1.0, black_level_val=-5, balance_ratio_val=3.4)
        set_frame_rate(cam, target_fps=120.0)

        cam.BeginAcquisition()
    
    gc.collect()
    
    while True:
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
        track.process_frame(img, top_down_view=True, printing=False)
    
    cam.EndAcquisition()
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
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--load", type=str2bool, default=False)
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
    #try:
    cam.Init()
    set_realtime_priority()
    
    check_isolation_status()
    
    configure_buffer_handling(cam)

    object_loc(cam, args.load)

    #except Exception as ex: #PySpin.SpinnakerException as ex:
    #    print("Error: %s" % ex)
    #finally:
    cam.EndAcquisition()
    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
