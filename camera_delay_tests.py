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
import serial
import struct

def set_realtime_priority():
    os.sched_setaffinity(0, {2,3})
    SCHED_FIFO = 1
    sched_param = ctypes.c_int(95)
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
            if min_val <= 1:
                buffer_count.SetValue(1)
                print("Buffer count set to 1")
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

def configure_camera(cam, gain_val=30.0):
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
        exposure_time.SetValue(100.0)
    else:
        print("Unable to set ExposureTime.")

    gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsAvailable(gain) and PySpin.IsWritable(gain):
        gain.SetValue(gain_val)
    else:
        print("Unable to set Gain.")
        
    # Try to disable image processing features that add latency
    processing_features = ["GammaEnable", "SharpnessEnable", "SaturationEnable", 
                          "HueEnable", "DefectCorrectStaticEnable"]
    
    for feature in processing_features:
        try:
            feature_node = PySpin.CBooleanPtr(nodemap.GetNode(feature))
            if PySpin.IsAvailable(feature_node) and PySpin.IsWritable(feature_node):
                feature_node.SetValue(False)
                print(f"{feature} disabled")
        except:
            print("Unable to disable " + feature)
            
    throughput_limit = PySpin.CIntegerPtr(nodemap.GetNode("DeviceLinkThroughputLimit"))
    if PySpin.IsAvailable(throughput_limit) and PySpin.IsWritable(throughput_limit):
        max_value = throughput_limit.GetMax()
        throughput_limit.SetValue(max_value)
        print(f"Set throughput limit to {max_value}")
    else:
        print("Unable to read or write throughput limit mode.")


def set_pixel_format(cam):
    nodemap = cam.GetNodeMap()
    
    pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    if PySpin.IsAvailable(pixel_format) and PySpin.IsWritable(pixel_format):
        # Try Mono8 first (fastest)
        format_entry = pixel_format.GetEntryByName("BayerRG8")
        if PySpin.IsAvailable(format_entry) and PySpin.IsReadable(format_entry):
            pixel_format.SetIntValue(format_entry.GetValue())
            print("Pixel format set to BayerRG8")
            
        else:
            print("unable to set pixel format to BayerRG8")
    else:
        print("Unable to read pixel format")
        
    ISP = PySpin.CBooleanPtr(nodemap.GetNode("IspEnable"))
    if PySpin.IsAvailable(ISP) and PySpin.IsWritable(ISP):
        ISP.SetValue(False)
    else:
        print("Unable to turn off ISP")
        
    print("Unable to set optimal pixel format")
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

def precise_timing_measurement(cam, N_TESTS=500):
    """Optimized timing measurement with minimal overhead"""
    
    # Pre-allocate arrays
    times = np.zeros(N_TESTS, dtype=np.float64)
    threshold = 100
    
    # Disable garbage collection during measurement

    
    try:
        cam.BeginAcquisition()
        PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
        BAUD = 460800
        
        img_shape = (1536, 1296)

        # === CONNECT ===
        ser = serial.Serial(PORT, BAUD, timeout=1)
        time.sleep(2)  # Wait for serial connection to settle
        
        # Warm up - discard first few frames
        for _ in range(10):
            image_result = cam.GetNextImage()
            img = image_result.GetData().reshape(img_shape)
            image_result.Release()
        print(img_shape)
        #for i in range(len(img)):
        #    for j in range(len(img[0])):
        #        if img[i,j] > 100:
        #            print((i,j))
        center = [915, 700]
        #return
        
        high = False
        for i in range(N_TESTS):
            while True:
                image_result = cam.GetNextImage()
                img = image_result.GetData().reshape(img_shape)
                image_result.Release()
                if img[center[0], center[1]] > threshold and not high:
                    ser.write(b'P')
                    data = ser.read(4)
                    if len(data) == 4:
                        (elapsed_time_ms,) = struct.unpack('<f', data)  # little-endian float
                        times[i] = elapsed_time_ms / 1000
                        high = True
                        break
                    else:
                        print("Timeout or invalid data")
                        return
                elif img[center[0], center[1]] < threshold:
                    high = False
        
        cam.EndAcquisition()
    except Exception as e:
        print("err")
        print(e)
        
    times = times[1:]
    print(times)
    
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
    
    # Distribution analysis
    ranges = [(0, 10), (10,20), (20,60), (60,100)]
    for min_r, max_r in ranges:
        count = np.sum((times >= min_r) & (times < max_r))
        percentage = (count / len(times)) * 100
        print(f"{min_r:.1f}-{max_r:.1f}ms: {count} frames ({percentage:.1f}%)")



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
        configure_camera(cam)
        set_pixel_format(cam)
        set_roi(cam,1296,1536,376,0)
        set_frame_rate(cam)

        precise_timing_measurement(cam)

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
