import torch
import PySpin
import cv2
import numpy as np
import sys
import os
import agent_processing as ap
import tracker
import time
from collections import deque

import multiprocessing as mp
from multiprocessing import shared_memory
import psutil
import serial

import threading
import queue

def configure_buffer_handling(cam):
    # Get TLStream nodemap (this is different from the regular nodemap)
    nodemap_tldevice = cam.GetTLStreamNodeMap()

    # Retrieve the buffer handling mode node
    buffer_handling_mode = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode("StreamBufferHandlingMode"))
    if PySpin.IsAvailable(buffer_handling_mode) and PySpin.IsWritable(buffer_handling_mode):
        # Set the buffer handling mode to NewestOnly
        newest_only = buffer_handling_mode.GetEntryByName("NewestOnly")
        if PySpin.IsAvailable(newest_only) and PySpin.IsReadable(newest_only):
            buffer_handling_mode.SetIntValue(newest_only.GetValue())
            print("Buffer Handling Mode set to NewestOnly")
        else:
            print("NewestOnly option not available")
    else:
        print("Unable to set Buffer Handling Mode")

    try:
        buffer_count = PySpin.CIntegerPtr(nodemap_tldevice.GetNode("StreamBufferCountManual"))
        if PySpin.IsAvailable(buffer_count) and PySpin.IsWritable(buffer_count):
            buffer_count.SetValue(1)  # Minimum buffers for lowest latency
            print("Buffer count set to 1")
    except:
        pass
    
    # Try setting buffer size mode
    try:
        buffer_size_mode = PySpin.CEnumerationPtr(nodemap_tldevice.GetNode("StreamBufferCountMode"))
        if PySpin.IsAvailable(buffer_size_mode) and PySpin.IsWritable(buffer_size_mode):
            manual_mode = buffer_size_mode.GetEntryByName("Manual")
            if PySpin.IsAvailable(manual_mode):
                buffer_size_mode.SetIntValue(manual_mode.GetValue())
                print("Buffer size mode set to Manual")
    except:
        pass

def configure_camera(cam, gain_val=30.0):
    # Get the camera node map
    nodemap = cam.GetNodeMap()
    
    # Disable automatic exposure and set exposure to 100 µs
    exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode("ExposureAuto"))
    if PySpin.IsAvailable(exposure_auto) and PySpin.IsWritable(exposure_auto):
        exposure_auto_off = exposure_auto.GetEntryByName("Off")
        exposure_auto.SetIntValue(exposure_auto_off.GetValue())
    else:
        print("Unable to disable ExposureAuto.")

    exposure_time = PySpin.CFloatPtr(nodemap.GetNode("ExposureTime"))
    if PySpin.IsAvailable(exposure_time) and PySpin.IsWritable(exposure_time):
        # ExposureTime is in microseconds
        exposure_time.SetValue(100.0)
    else:
        print("Unable to set ExposureTime.")

    # Disable automatic gain and set gain to 40
    gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode("GainAuto"))
    if PySpin.IsAvailable(gain_auto) and PySpin.IsWritable(gain_auto):
        gain_auto_off = gain_auto.GetEntryByName("Off")
        gain_auto.SetIntValue(gain_auto_off.GetValue())
    else:
        print("Unable to disable GainAuto.")

    gain = PySpin.CFloatPtr(nodemap.GetNode("Gain"))
    if PySpin.IsAvailable(gain) and PySpin.IsWritable(gain):
        gain.SetValue(gain_val)
    else:
        print("Unable to set Gain.")

def set_pixel_format(cam, format_type="mono"):
    """
    Set the pixel format for the camera.
    
    Args:
        cam: PySpin camera object
        format_type: "mono" for grayscale, "red", "green", or "blue" for single channels
    """
    nodemap = cam.GetNodeMap()
    
    # Get the pixel format node
    pixel_format = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
    
    if not PySpin.IsAvailable(pixel_format) or not PySpin.IsWritable(pixel_format):
        print("Unable to access PixelFormat node")
        return False
    
    # Define format mappings
    format_map = {
        "mono": "Mono8",          # 8-bit grayscale - fastest option
        "red": "BayerRG8",        # You'd extract red channel from Bayer
        "green": "BayerRG8",      # You'd extract green channel from Bayer  
        "blue": "BayerRG8"        # You'd extract blue channel from Bayer
    }
    
    # For true single channel, Mono8 is your best bet
    if format_type == "mono":
        target_format = "Mono8"
    else:
        # For color cameras, you might need to use RGB8 and extract channels in software
        target_format = "RGB8"
    
    # Get the desired format entry
    format_entry = pixel_format.GetEntryByName(target_format)
    
    if PySpin.IsAvailable(format_entry) and PySpin.IsReadable(format_entry):
        pixel_format.SetIntValue(format_entry.GetValue())
        print(f"Pixel format set to {target_format}")
        return True
    else:
        print(f"Format {target_format} not available. Available formats:")
        # List available formats
        for i in range(pixel_format.GetNumEntries()):
            entry = pixel_format.GetEntryByIndex(i)
            if PySpin.IsAvailable(entry):
                print(f"  - {entry.GetSymbolic()}")
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

def disable_new_sdk_features(cam):
    """Disable features that might add timing jitter"""
    nodemap = cam.GetNodeMap()
    
    # Disable any new image processing features
    features_to_disable = [
        "ImageCompressionMode",
        "ImageCompressionQuality", 
        "ChunkModeActive",
        "EventSelector",
        "LogicBlockSelector",
        "CounterSelector",
        "TimerSelector",
        "EncoderSelector",
        "LineSelector"
    ]
    
    for feature in features_to_disable:
        try:
            node = nodemap.GetNode(feature)
            if PySpin.IsAvailable(node) and PySpin.IsWritable(node):
                if node.GetPrincipalInterfaceType() == PySpin.intfIBoolean:
                    bool_node = PySpin.CBooleanPtr(node)
                    bool_node.SetValue(False)
                    print(f"Disabled {feature}")
                elif node.GetPrincipalInterfaceType() == PySpin.intfIEnumeration:
                    enum_node = PySpin.CEnumerationPtr(node)
                    try:
                        off_entry = enum_node.GetEntryByName("Off")
                        if PySpin.IsAvailable(off_entry):
                            enum_node.SetIntValue(off_entry.GetValue())
                            print(f"Set {feature} to Off")
                    except:
                        pass
        except:
            pass

def precise_timing_measurement(cam, num_frames=1000):

    """More accurate timing measurement using hardware timestamps if available"""
    
    # Try to enable timestamp
    nodemap = cam.GetNodeMap()
    try:
        timestamp_latch = PySpin.CEnumerationPtr(nodemap.GetNode("TimestampLatch"))
        if PySpin.IsAvailable(timestamp_latch) and PySpin.IsWritable(timestamp_latch):
            timestamp_latch_value = timestamp_latch.GetEntryByName("Start")
            if PySpin.IsAvailable(timestamp_latch_value):
                timestamp_latch.SetIntValue(timestamp_latch_value.GetValue())
                print("Hardware timestamp enabled")
    except:
        pass

    cam.BeginAcquisition()
    
    # Collect both system time and hardware timestamps
    system_times = []
    hw_timestamps = []
    frame_intervals = []
    
    prev_system_time = None
    prev_hw_timestamp = None

    image_result = cam.GetNextImage()
    img = image_result.GetNDArray()
    image_result.Release()
    print(img.shape)

    for _ in range(50):
        image_result = cam.GetNextImage()
        image_result.Release()

    

    for i in range(num_frames):
        """
        system_time = time.perf_counter()
        image_result = cam.GetNextImage()
        
        # Get hardware timestamp if available
        hw_timestamp = None
        try:
            hw_timestamp = image_result.GetTimeStamp()
        except:
            pass
        img = image_result.GetNDArray()
        image_result.Release()
        
        if i > 0:  # Skip first frame
            system_interval = (system_time - prev_system_time) * 1000
            frame_intervals.append(system_interval)
            
            if hw_timestamp and prev_hw_timestamp:
                # Camera timestamps are usually in nanoseconds
                hw_interval = (hw_timestamp - prev_hw_timestamp) / 1_000_000  # Convert to ms
                hw_timestamps.append(hw_interval)
        
        system_times.append(system_time)
        
        
        prev_system_time = system_time
        prev_hw_timestamp = hw_timestamp
        """
        prev_time = time.time()
        while time.time() < prev_time + 0.005:
            x = 1251/23.403
        system_interval = ((time.time() - prev_time) * 1000)
        frame_intervals.append(system_interval)
    
    cam.EndAcquisition()
    
    # Analyze timing
    if frame_intervals:
        avg_interval = sum(frame_intervals) / len(frame_intervals)
        std_dev = (sum((x - avg_interval) ** 2 for x in frame_intervals) / len(frame_intervals)) ** 0.5
        min_interval = min(frame_intervals)
        max_interval = max(frame_intervals)
        
        print(f"\n=== System Timing Analysis ===")
        print(f"Average interval: {avg_interval:.3f}ms")
        print(f"Standard deviation: {std_dev:.3f}ms")
        print(f"Min interval: {min_interval:.3f}ms")
        print(f"Max interval: {max_interval:.3f}ms")
        print(f"Jitter range: {max_interval - min_interval:.3f}ms")
        print(f"Actual FPS: {1000/avg_interval:.2f}")
        
        # Print distribution
        ranges = [(0,7),(7,7.5),(7.5, 8.0), (8.0, 8.5), (8.5, 9.0), (9.0, 9.5), (9.5, 10.0), (10.0, 100)]
        for min_r, max_r in ranges:
            count = sum(1 for x in frame_intervals if min_r <= x < max_r)
            percentage = (count / len(frame_intervals)) * 100
            print(f"{min_r:.1f}-{max_r:.1f}ms: {count} frames ({percentage:.1f}%)")
    
    if hw_timestamps:
        hw_avg = sum(hw_timestamps) / len(hw_timestamps)
        hw_std = (sum((x - hw_avg) ** 2 for x in hw_timestamps) / len(hw_timestamps)) ** 0.5
        print(f"\n=== Hardware Timing Analysis ===")
        print(f"HW Average interval: {hw_avg:.3f}ms")
        print(f"HW Standard deviation: {hw_std:.3f}ms")
        print(len(hw_timestamps))
        print(f"min: {np.min(hw_timestamps)}")
        print(f"max: {np.max(hw_timestamps)}")


def main():
    current_process = psutil.Process()
    current_process.cpu_affinity([6,7,8,9,10,11])

    current_process.nice(psutil.HIGH_PRIORITY_CLASS)
    #current_process.nice(psutil.REALTIME_PRIORITY_CLASS)

    system = PySpin.System.GetInstance()
    
    # Retrieve list of cameras from the system
    cam_list = system.GetCameras()
    num_cameras = cam_list.GetSize()
    
    if num_cameras == 0:
        print("No cameras detected!")
        cam_list.Clear()
        system.ReleaseInstance()
        sys.exit(1)
    
    # Select the first camera
    cam = cam_list.GetByIndex(0)
    try:
        cam.Init()
        configure_buffer_handling(cam)
        configure_camera(cam)
        while not set_pixel_format(cam, "mono"):
            pass
        set_roi(cam,64,64)
        disable_new_sdk_features(cam)
        set_frame_rate(cam)

        prev_time = None

        precise_timing_measurement(cam)
        """
        cam.BeginAcquisition()
        while True:
            image_result = cam.GetNextImage()
            if prev_time is None:
                prev_time = time.time()
            image_data = image_result.GetNDArray()
            image_result.Release()
            time_save = time.time()
            #if time_save - prev_time >  0.013:
            print(time_save - prev_time)
            prev_time = time_save

        cam.EndAcquisition()
        cam.DeInit()
        """
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
