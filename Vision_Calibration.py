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

table_bounds = np.array([1.993, 0.992])

mallet_r = 0.1011 / 2 #0.0508
puck_r = 0.0636 / 2
margin_bounds = 0.0

num_points = 11

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
            buffer_count.SetValue(20)
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
    
    ser.reset_input_buffer()
    
    # Read entire buffer
    while ser.in_waiting < (num_points+1)*11-1:
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

def begin_calibrations(cam):
    """Optimized timing measurement with minimal overhead"""

    j=3

    PORT = '/dev/ttyUSB0'  # Adjust this to COM port or /dev/ttyUSBx
    BAUD = 460800

    # === CONNECT ===
    ser = serial.Serial(PORT, BAUD, timeout=0)
    
    time.sleep(1)
    img_shape = (1536, 1296)
    
    set_pixel_format(cam, mode="Mono8")
    configure_camera(cam, gain_val=1.0, exposure_val=10000.0)
    set_frame_rate(cam, target_fps=20.0)
    
    cam.BeginAcquisition()
    
    # Warm up - discard first few fra
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
    while not setup.see_aruco_pixels(img):
        cv2.imshow("arucos", img[::2, ::2])
        cv2.waitKey(0)
        image = cam.GetNextImage()
        img = np.clip(image.GetData().reshape(img_shape) * bright_filter, 0, 255).astype(np.uint8)
        image.Release()
        
        
    np.save(f"img_data_{j}.npy", np.array(img))
        
    cv2.destroyAllWindows()
    
    cam.EndAcquisition()
                                  
    set_pixel_format(cam, mode="BayerRG8")
    configure_camera(cam, gain_val=33.0, exposure_val=100.0)
    set_frame_rate(cam, target_fps=120.0)
    
    print("Remove mallet + puck from view")
    input()
    
    cam.BeginAcquisition()
    
    image = cam.GetNextImage()
    img = image.GetData().reshape(img_shape)
    image.Release()
    
    y_max = np.max(img[:, int(img.shape[1]/2-30):int(img.shape[1]/2+30)], axis=1)
    cutoff = np.array([np.max(y_max[max(0,i-30):min(i+30, len(y_max))]) for i in range(len(y_max))])
    
    #cutoff_thres = np.minimum(np.tile(cutoff[:,None], (1,1296)), 225) + 25
    #cv2.imshow("thresh", cutoff_thres[::2, ::2])
    #cv2.imshow("img", img[::2, ::2])
    #cv2.waitKey(0)
    print("Move mallet up or down")
    input()
    
    del y_max
    
    for _ in range(10):
        image = cam.GetNextImage()
        img = image.GetData().reshape(img_shape)
        image.Release()
    
    y_max2 = np.max(img[:, int(img.shape[1]/2-30):int(img.shape[1]/2+30)], axis=1)
    cutoff2 = np.array([np.max(y_max2[max(0,i-30):min(i+30, len(y_max2))]) for i in range(len(y_max2))])
    
    #cutoff_thres = np.minimum(np.tile(cutoff2[:,None], (1,1296)), 225) + 25
    #cv2.imshow("thresh", cutoff_thres[::2, ::2])
    #cv2.imshow("img", img[::2, ::2])
    #cv2.waitKey(0)
    
    cutoff = np.maximum(cutoff, cutoff2)
    #cutoff_thres = np.minimum(np.tile(cutoff[:,None], (1,1296)), 225) + 25
    #cv2.imshow("thresh", cutoff_thres[::2, ::2])
    #cv2.waitKey(0)
    
    del y_max2
    del cutoff2
    
    setup.set_thresh_map(cutoff)
    
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
    
    ser.reset_input_buffer()
    
    input("Remove calibration device")
    
    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.reset_input_buffer()
    
    input("Don't Turn on Power to Motors")

    ser.write(b'\n')
    
    while ser.in_waiting == 0:
        continue
    ser.reset_input_buffer()

    
    input("Enter to Start")
    time.sleep(1.0)
    
    gc.collect()
    
    ser.reset_input_buffer()
    
    while ser.in_waiting < (num_points+1) * 11:
        pass
    
    for _ in range(11):
        get_mallet(ser)
    
    target_puck_points = []
    for x in [0.035+puck_r, 0.01+puck_r+0.199, 0.01+puck_r+2*0.199, 0.01+puck_r+3*0.199, 0.01+puck_r+4*0.199, 0.01+puck_r+5*0.199, 0.01+puck_r+7*0.199, 0.01+puck_r+9*0.199]:
        for y in [puck_r, 0.199-0.04+puck_r, 2*0.199-0.04+puck_r]:
            target_puck_points.append([x,y])
            
    for x in [0.035+puck_r, 0.01+puck_r+0.199, 0.01+puck_r+2*0.199, 0.01+puck_r+3*0.199, 0.01+puck_r+4*0.199, 0.01+puck_r+5*0.199, 0.01+puck_r+7*0.199, 0.01+puck_r+9*0.199]:
        for y in [table_bounds[1]-puck_r, table_bounds[1]-0.199+0.04-puck_r, table_bounds[1]-2*0.199+0.04-puck_r]:
            target_puck_points.append([x,y])
    
    gc.collect()
    
    #pxls = np.zeros((8*6,2))
    #locations = np.zeros((8*6, 2))
    
    pxls = np.load(f"pxls_data_{j}.npy")
    locations = np.load(f"location_data_{j}.npy")
    
    pxls[39,:] = 0
    locations[39,:] = 0
    
    gc.collect()
    
    print("START")

    for idx, target_puck in enumerate(target_puck_points):
        if pxls[idx, 0] != 0:
            continue
        mp = np.array([target_puck[0], target_puck[1]])
        if target_puck[0] < 0.4:
            mp[0] += 0.199 - 0.04 + puck_r + mallet_r
            if abs(target_puck[1] - 0.992/2) > 0.4:
                mp[0] += 0.04
        else:
            mp[0] -= 0.199 - 0.04 + puck_r + mallet_r
            if abs(target_puck[1] - 0.992/2) > 0.4:
                mp[0] -= 0.04
        
        if target_puck[0] > 1.2 and target_puck[0] < 1.6:
            mp[0] -= 0.199
        elif target_puck[0] > 1.6:
            mp[0] -= 3*0.199
            
        if (target_puck[1] > 0.5 and abs(target_puck[1] - 0.992/2) > 0.4) or (target_puck[1] < 0.5 and abs(target_puck[1] - 0.992/2) < 0.4):
            mp[1] -= 0.199 / 2
        else:
            mp[1] += 0.199/2
            
        print((abs(target_puck[0] - mp[0]) - puck_r - mallet_r) / 0.199)
        
        while True:
            close_enough = False
            print("A1")
            get_mallet(ser)
            pos, vel, acc = get_init_conditions()
            print("B1")
                
            top_down_image = np.ones((int(table_bounds[1] * 500), int(table_bounds[0] * 500), 3), dtype=np.uint8) * 255
            
            x_img = int(mp[0] * 500)  # scale factor for x
            y_img = int(mp[1] * 500)  # invert y-axis for display
            cv2.circle(top_down_image, (x_img, y_img), int(mallet_r * 500), (255, 255, 0), -1)
            
            x_img = int(target_puck[0] * 500)  # scale factor for x
            y_img = int(target_puck[1] * 500)  # invert y-axis for display
            cv2.circle(top_down_image, (x_img, y_img), int(puck_r * 500), (0, 0, 0), -1)
            
            x_img = int(pos[0] * 500)  # scale factor for x
            y_img = int(pos[1] * 500)  # invert y-axis for display
            #print(pos)
            #print('--')
            #print(pos - mp)
            passed = False
            if np.linalg.norm(pos - mp) < 0.01:
                close_enough = True
                cv2.circle(top_down_image, (x_img, y_img), int(mallet_r * 500), (255, 0, 0), -1)
                passed = True
            else:
                cv2.circle(top_down_image, (x_img, y_img), int(mallet_r * 500), (100, 0, 100), -1)
            
            print("C1")
            img_passed = False
            while not img_passed:
                try:
                    image = cam.GetNextImage(1000)
                    
                    if image.IsIncomplete():
                        print(f"Image incomplete: {image.GetImageStatus()}")
                        image.Release()
                        raise Exception("Incomplete image")
                        
                    print("D1")
                    img = image.GetNDArray().reshape(img_shape)
                    img_passed=True
                    image.Release()
                except Exception as e:
                    print(e)
                    cam.EndAcquisition()
                    
                    time.sleep(0.5)
                    
                    cam.BeginAcquisition()
                    
            print("D3")
            
            pxl = setup.get_puck_pixel(img)
            print("D4")
            
            if (pxl is not None) and passed:
                print("A")
                img_np = np.array(img)
                cv2.circle(img_np, (int(pxl[0]), int(pxl[1])), 20, (0, 255, 0), -1)
                cv2.imshow("puxk", img_np[::2, ::2])
                cv2.waitKey(1)
                time.sleep(5)
                print("B")
                img_passed = False
                while not img_passed:
                    try:
                        image = cam.GetNextImage(1000)
                        
                        if image.IsIncomplete():
                            print(f"Image incomplete: {image.GetImageStatus()}")
                            image.Release()
                            raise Exception("Incomplete image")
                            
                        print("B3")
                        img = image.GetNDArray().reshape(img_shape)
                        img_passed=True
                        image.Release()
                    except Exception as e:
                        print(e)
                        cam.EndAcquisition()
                        
                        time.sleep(0.5)
                        
                        cam.BeginAcquisition()

                image.Release()
                pxl = setup.get_puck_pixel(img)
                print(pxl)
                if pxl is not None:
                    print("A2")
                    get_mallet(ser)
                    pos, vel, acc = get_init_conditions()
                    print("B2")
                    pxls[idx] = pxl
                    location = np.array([table_bounds[0] - (pos[0] + target_puck[0]-mp[0]), table_bounds[1] - target_puck[1]])
                    locations[idx] = location
                    
                    np.save(f"pxls_data_{j}.npy", np.array(pxls))  
                    np.save(f"location_data_{j}.npy", np.array(locations))
                    
                    img_np = np.array(img)
                    cv2.circle(img_np, (int(pxl[0]), int(pxl[1])), 20, (0, 255, 0), -1)
                    cv2.imshow("puxk", img_np[::2, ::2])
                    cv2.waitKey(1)
                    break
            
            cv2.imshow("top_down_table", top_down_image)
            print("E1")
            cv2.waitKey(1)
            print("F1") 
    
    cam.EndAcquisition()


def main():
    
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
    
    configure_buffer_handling(cam)
    set_roi(cam,1296,1536,376,0)

    begin_calibrations(cam)

    cam.DeInit()
    del cam
    cam_list.Clear()
    system.ReleaseInstance()

if __name__ == "__main__":
    main()
