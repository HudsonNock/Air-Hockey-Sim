#!/usr/bin/env python3
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

VISION_BUFFER_SIZE = 14  # Keep last 20 vision frames
MALLET_BUFFER_SIZE = 14  # Keep last 20 mallet states
SHARED_DATA_SIZE = 1024  # Size in bytes for each shared memory block

class SharedVisionBuffer:
    """Manages shared memory for vision data (puck and opponent mallet positions)"""
    def __init__(self):
        # Create shared memory for vision data
        self.shm = shared_memory.SharedMemory(create=True, size=SHARED_DATA_SIZE, name='vision_buffer')
        self.lock = mp.Lock()
        
        # Initialize buffer structure: [write_index, count, data...]
        # data format: timestamp, puck_pos[2], op_mallet_pos[2] = 5 floats per entry
        buffer_array = np.ndarray((1 + 1 + VISION_BUFFER_SIZE * 5,), dtype=np.float64, buffer=self.shm.buf)
        buffer_array[0] = 0  # write_index
        buffer_array[1] = 0  # count

        for i in range(len(buffer_array)):
            buffer_array[i] = 0
        
    def write(self, timestamp, puck_pos, op_mallet_pos):
        with self.lock:
            buffer_array = np.ndarray((1 + 1 + VISION_BUFFER_SIZE * 5,), dtype=np.float64, buffer=self.shm.buf)
            write_idx = int(buffer_array[0])
            count = int(buffer_array[1])
            
            # Write data at current index
            start_idx = 2 + write_idx * 5
            buffer_array[start_idx] = timestamp
            buffer_array[start_idx + 1:start_idx + 3] = puck_pos
            buffer_array[start_idx + 3:start_idx + 5] = op_mallet_pos
            
            # Update indices
            buffer_array[0] = (write_idx + 1) % VISION_BUFFER_SIZE
            buffer_array[1] = min(count + 1, VISION_BUFFER_SIZE)
    
    def read(self, indices=[0,1,2,5,11]):
        """Read n_recent most recent entries plus historical entries at specified indices"""
        with self.lock:
            buffer_array = np.ndarray((1 + 1 + VISION_BUFFER_SIZE * 5,), dtype=np.float64, buffer=self.shm.buf)
            write_idx = int(buffer_array[0])
            
            data = []
            
            # Get recent data
            for i in indices:
                idx = (write_idx - 1 - i) % VISION_BUFFER_SIZE
                start_idx = 2 + idx * 5
                #entry = {
                #    'timestamp': buffer_array[start_idx],
                #    'puck_pos': buffer_array[start_idx + 1:start_idx + 3].copy(),
                #    'op_mallet_pos': buffer_array[start_idx + 3:start_idx + 5].copy()
                #}
                data.append(buffer_array[start_idx:start_idx+5].copy())
            
            return data
    
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()

class VisionBuffer:

    def __init__(self):
        # Initialize buffer structure: [write_index, count, data...]
        # data format: timestamp, puck_pos[2], op_mallet_pos[2] = 5 floats per entry
        self.buffer_array = np.zeros((VISION_BUFFER_SIZE * 5,), dtype=np.float64)
        self.write_idx = 0
        self.count = 0
        
    def write(self, timestamp, puck_pos, op_mallet_pos):
        # Write data at current index
        start_idx = 2 + self.write_idx * 5
        self.buffer_array[start_idx] = timestamp
        self.buffer_array[start_idx + 1:start_idx + 3] = puck_pos
        self.buffer_array[start_idx + 3:start_idx + 5] = op_mallet_pos
        
        # Update indices
        self.write_idx = (self.write_idx + 1) % VISION_BUFFER_SIZE
        self.count = min(self.count + 1, VISION_BUFFER_SIZE)
    
    def read(self, indices=[0,1,2,5,11]):
        """Read n_recent most recent entries plus historical entries at specified indices"""
            
        data = []
        
        # Get recent data
        for i in indices:
            idx = (self.write_idx - 1 - i) % VISION_BUFFER_SIZE
            start_idx = 2 + idx * 5
            #entry = {
            #    'timestamp': buffer_array[start_idx],
            #    'puck_pos': buffer_array[start_idx + 1:start_idx + 3].copy(),
            #    'op_mallet_pos': buffer_array[start_idx + 3:start_idx + 5].copy()
            #}
            data.append(self.buffer_array[start_idx:start_idx+5].copy())
        
        return data

class SharedMalletBuffer:
    """Manages shared memory for mallet state data"""
    def __init__(self):
        self.shm = shared_memory.SharedMemory(create=True, size=SHARED_DATA_SIZE, name='mallet_buffer')
        self.lock = mp.Lock()
        
        # Initialize buffer: [write_index, count, data...]
        # data format: timestamp, pos[2], vel[2], acc[2] = 7 floats per entry
        buffer_array = np.ndarray((1 + 1 + 7 + MALLET_BUFFER_SIZE * 3,), dtype=np.float64, buffer=self.shm.buf)
        buffer_array[0] = 0  # write_index
        buffer_array[1] = 0  # count

        for i in range(len(buffer_array)):
            buffer_array[i] = 0
        
        
    def write(self, timestamp, pos, vel, acc, buffer):
        with self.lock:
            buffer_array = np.ndarray((1 + 1 + MALLET_BUFFER_SIZE * 7,), dtype=np.float64, buffer=self.shm.buf)
            buffer_array[2] = timestamp
            buffer_array[3:5] = pos
            buffer_array[5:7] = vel
            buffer_array[7:9] = acc

            if buffer:
                write_idx = int(buffer_array[0])
                count = int(buffer_array[1])
                
                start_idx = 9 + write_idx * 3
                buffer_array[start_idx] = timestamp
                buffer_array[start_idx + 1:start_idx + 3] = pos
                
                buffer_array[0] = (write_idx + 1) % MALLET_BUFFER_SIZE
                buffer_array[1] = min(count + 1, MALLET_BUFFER_SIZE)
    
    def read(self, latest, indices=None):
        """Read the most recent mallet state"""
        with self.lock:
            buffer_array = np.ndarray((1 + 1 + MALLET_BUFFER_SIZE * 7,), dtype=np.float64, buffer=self.shm.buf)
            write_idx = int(buffer_array[0])
            count = int(buffer_array[1])
            
            if count == 0:
                return None
            
            data = []

            if latest:
                data.append(buffer_array[2:9])
            
            if indices is not None:
            # Get recent data
                for i in indices:
                    idx = (write_idx - 1 - i) % MALLET_BUFFER_SIZE
                    start_idx = 9 + idx * 3
                    data.append(buffer_array[start_idx:start_idx+3].copy())
            
            return data
    
    def cleanup(self):
        self.shm.close()
        self.shm.unlink()

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

def set_frame_rate(cam):
    # Get the camera node map
    nodemap = cam.GetNodeMap()
    
    # Enable frame rate control
    frame_rate_enable = PySpin.CBooleanPtr(nodemap.GetNode("AcquisitionFrameRateEnable"))
    if PySpin.IsAvailable(frame_rate_enable) and PySpin.IsWritable(frame_rate_enable):
        frame_rate_enable.SetValue(True)
    else:
        print("Unable to enable AcquisitionFrameRate.")

    # Set the frame rate to the maximum allowed
    frame_rate = PySpin.CFloatPtr(nodemap.GetNode("AcquisitionFrameRate"))
    if PySpin.IsAvailable(frame_rate) and PySpin.IsWritable(frame_rate):
        max_frame_rate = frame_rate.GetMax()
        frame_rate.SetValue(max_frame_rate)
        print("AcquisitionFrameRate set to maximum: {:.2f} fps".format(max_frame_rate))
    else:
        print("Unable to set AcquisitionFrameRate.")

def get_external_matrix(cam):
    # Begin image acquisition
    configure_gain(cam, gain_val=45.0)
    cam.BeginAcquisition()
    print("External started...")
    
    try:
        while True:
            # Retrieve next image with a timeout of 1000ms
            image_result = cam.GetNextImage()
            #if image_result.IsIncomplete():
            #    print("Image incomplete with image status %d ..." % image_result.GetImageStatus())
            #else:
                # Convert image to a NumPy array (BGR or grayscale depending on your camera configuration)
            image_data = image_result.GetNDArray()

            image_result.Release()

            if setup.run_extrinsics(image_data):
                break
    except KeyboardInterrupt:
        print("Image acquisition interrupted by user.")
    finally:
        cam.EndAcquisition()
        print("External ended.")

"""
def image_process(cam, Normal_Mallet, rotation_matrix, translation_vector, z_pixel_map):
    current_process = psutil.Process()
    current_process.cpu_affinity([2, 3, 4, 5])

    current_process.nice(psutil.HIGH_PRIORITY_CLASS)

    if Normal_Mallet:
        op_mallet_z = (120.94)*10**(-3)
    else:
        op_mallet_z = 70.44 * 10**(-3)

    vision_shm = shared_memory.SharedMemory(name='vision_buffer')
    vision_buffer = SharedVisionBuffer.__new__(SharedVisionBuffer)
    vision_buffer.shm = vision_shm
    vision_buffer.lock = mp.Lock()

    camera_tracker = tracker.CameraTracker(rotation_matrix, translation_vector, z_pixel_map, op_mallet_z)

    try:
        while True:
            image_result = cam.GetNextImage()
            image_data = image_result.GetNDArray()
            image_result.Release()
            time_save = time.time()

            puck_pos, op_mallet_pos = camera_tracker.process_frame(image_data)
            puck_pos = np.array([ap.height - puck_pos[0], ap.width - puck_pos[1]])
            op_mallet_pos = np.array([ap.height - op_mallet_pos[0], ap.width - op_mallet_pos[1]])
            vision_buffer.write(time_save, puck_pos, op_mallet_pos)
    except KeyboardInterrupt:
        pass
    vision_shm.close()
"""

def agent_process(cam, Normal_Mallet, rotation_matrix, translation_vector, z_pixel_map, writeQueue):

    current_process = psutil.Process()
    current_process.cpu_affinity([2,3,4,5,6])

    current_process.nice(psutil.HIGH_PRIORITY_CLASS)

    if Normal_Mallet:
        op_mallet_z = (120.94)*10**(-3)
    else:
        op_mallet_z = 70.44 * 10**(-3)

    #vision_shm = shared_memory.SharedMemory(name='vision_buffer')
    mallet_shm = shared_memory.SharedMemory(name='mallet_buffer')
    
    #vision_buffer = SharedVisionBuffer.__new__(SharedVisionBuffer)
    #vision_buffer.shm = vision_shm
    #vision_buffer.lock = mp.Lock()
    vision_buffer = VisionBuffer()
    
    mallet_buffer = SharedMalletBuffer.__new__(SharedMalletBuffer)
    mallet_buffer.shm = mallet_shm
    mallet_buffer.lock = mp.Lock()

    camera_tracker = tracker.CameraTracker(rotation_matrix, translation_vector, z_pixel_map, op_mallet_z)

    cam.BeginAcquisition()

    while True:
        #vision_data = vision_buffer.read()
        image_result = cam.GetNextImage()
        image_data = image_result.GetNDArray()
        image_result.Release()
        time_save = time.time()

        puck_pos, op_mallet_pos = camera_tracker.process_frame(image_data)
        puck_pos = np.array([ap.height - puck_pos[0], ap.width - puck_pos[1]])
        op_mallet_pos = np.array([ap.height - op_mallet_pos[0], ap.width - op_mallet_pos[1]])
        vision_buffer.write(time_save, puck_pos, op_mallet_pos)
        vision_data = vision_buffer.read()
        mallet_data = mallet_buffer.read()
        for data in vision_data:
            for i in range(len(data)):
                if data[i] == 0:
                    continue
        for data in mallet_data:
            for i in range(len(data)):
                if data[i] == 0:
                    continue
        break

    writeQueue.put("got_init\n".encode())

    time.sleep(0.2)

    mallet_data = mallet_buffer.read(True)

    data = ap.update_path(mallet_data[1:3], mallet_data[3:5], mallet_data[5:7], [0.3,0.47], [5,5])
    writeQueue.put(b'\n' + data + b'\n')
    time.sleep(0.1)

    try:
        while True:
            image_result = cam.GetNextImage()
            image_data = image_result.GetNDArray()
            image_result.Release()
            time_save = time.time()

            puck_pos, op_mallet_pos = camera_tracker.process_frame(image_data)
            puck_pos = np.array([ap.height - puck_pos[0], ap.width - puck_pos[1]])
            op_mallet_pos = np.array([ap.height - op_mallet_pos[0], ap.width - op_mallet_pos[1]])
            vision_buffer.write(time_save, puck_pos, op_mallet_pos)

            vision_data = vision_buffer.read()
            mallet_data = mallet_buffer.read(True, indices=[3, 6, 9, 12])
            ap.take_action(vision_data, mallet_data, mallet_buffer, writeQueue)

            image_result = cam.GetNextImage()
            image_data = image_result.GetNDArray()
            image_result.Release()
            time_save = time.time()

            puck_pos, op_mallet_pos = camera_tracker.process_frame(image_data)
            puck_pos = np.array([ap.height - puck_pos[0], ap.width - puck_pos[1]])
            op_mallet_pos = np.array([ap.height - op_mallet_pos[0], ap.width - op_mallet_pos[1]])
            vision_buffer.write(time_save, puck_pos, op_mallet_pos)
    except KeyboardInterrupt:
        pass

    #vision_shm.close()
    mallet_shm.close()

def serial_io(ser, writeQueue):
    current_process = psutil.Process()
    current_process.cpu_affinity([7,8])    

    current_process.nice(psutil.HIGH_PRIORITY_CLASS)

    mallet_shm = shared_memory.SharedMemory(name='mallet_buffer')
    mallet_buffer = SharedMalletBuffer.__new__(SharedMalletBuffer)
    mallet_buffer.shm = mallet_shm
    mallet_buffer.lock = mp.Lock()

    past_time = time.time()
    try:
        while True:
            # Handle writes
            try:
                ser.write(writeQueue.get_nowait())
            except:
                pass

            # Handle reads
            if ser.in_waiting:
                pos, vel, acc, passed = ap.get_mallet(ser)
                if passed:
                    curr_time = time.time()
                    if curr_time - past_time > 0.01:
                        mallet_buffer.write(time.time(), pos, vel, acc, True)
                        past_time = curr_time
                    else:
                        mallet_buffer.write(time.time(), pos, vel, acc, False)
                        
    except KeyboardInterrupt:
        pass
    mallet_shm.close()

def init_processes(cam, Normal_Mallet):

    #vision_buffer = SharedVisionBuffer()
    mallet_buffer = SharedMalletBuffer()

    ser = serial.Serial('COM3', 460800, timeout=0)

    if ser.in_waiting:
        ser.read(ser.in_waiting).decode('utf-8')
    ser.write(b'1')

    ap.mallet_calibration(ser)
    ap.nn_mode(ser)

    writeQueue = mp.Queue()

    serial_process = mp.Process(target=serial_io, args=(ser, writeQueue))

    #camera_process = mp.Process(target=image_process, args=(cam,
    #                                                 Normal_Mallet,
    #                                                 setup.rotation_matrix,
    #                                                 setup.translation_vector,
    #                                                 setup.z_pixel_map))
    
    agent_process = mp.Process(target=agent_process, args=(cam,
                                                           Normal_Mallet,
                                                           setup.rotation_matrix,
                                                           setup.translation_vector,
                                                           setup.z_pixel_map,
                                                           writeQueue))

    serial_process.start()

    cam.BeginAcquisition()
    print("Camera acquisition started...")
    
    # Start the process
    #camera_process.start()
    agent_process.start()

    try:
        serial_process.join()
        #camera_process.join()
        agent_process.join()
    except KeyboardInterrupt:
        print("shutting down...")
        serial_process.terminate()
        #camera_process.terminate()
        agent_process.terminate()
    finally:
        #vision_buffer.cleanup()
        mallet_buffer.cleanup()

def get_calibration_data(cam,j):
    configure_gain(cam, gain_val=30.0)
    cam.BeginAcquisition()
    print("Beginning puck data collection: x axis")
    pxls = []
    i = 0
    while i < 5+4+5+4+15:
        if i < 5:
            print(f"Place puck on x axis | {i+1}/5 [Enter]")
        elif i < 5+4:
            print(f"Place puck on y axis | {i-5+1}/4")
        elif i < 5+4+5:
            print(f"Place puck on offset x axis | {i+1-5-4}/5 [Enter]")
        elif i < 5+4+5+4:
            print(f"Place puck on offset y axis | {i+1-5-4-5}/4 [Enter]")
        else:
            print(f"Place puck on circle {i+1-5-4-5-4} | [Enter]")
        input()
        image_result = cam.GetNextImage()
        image_data = image_result.GetNDArray()
        pxl = setup.get_puck_pixel(image_data)
        if pxl is None:
            i -= 1
            print("Can not detect puck, retry")
        else:
            pxls.append(pxl)
        i += 1

    np.save(f"puck_pxls_{j}.npy", np.array(pxls))

    cam.EndAcquisition()

    configure_gain(cam, gain_val=45.0)

    cam.BeginAcquisition()

    while True:
        print("Finding Aruco Points, Move Hand Away [Enter]")
        input()
        image_result = cam.GetNextImage()
        image_data = image_result.GetNDArray()
        if setup.see_aruco_pixels(image_data):
            np.save(f"img_data_{j}.npy", image_data)
            cam.EndAcquisition()
            print("success")
            return 
        print("Did not detect Aruco Markers, try again")

def getArucoPosSaved(n):
    pxls = np.load("puck_pxls_1.npy")
    image_data = np.load("img_data_1.npy")
    imgs = [image_data]

    for i in range(2, n+1):
        pxls_i = np.load(f"puck_pxls_{i}.npy")
        image_data_i = np.load(f"img_data_{i}.npy")
        pxls = np.vstack((pxls, pxls_i))
        imgs.append(image_data_i)

    puck_pxls_wall_x = np.zeros((n*10,2), dtype=np.float32)
    puck_pxls_wall_y = np.zeros((n*8,2), dtype=np.float32)
    puck_pxls_location = np.zeros((n*15, 2), dtype=np.float32)
    for i in range(n):
        puck_pxls_wall_x[10*i:10*i+5] = pxls[33*i:33*i+5]
        puck_pxls_wall_y[8*i:8*i+4] = pxls[33*i+5:33*i+9]
        puck_pxls_wall_x[10*i+5:10*i+10] = pxls[33*i+9:33*i+14]
        puck_pxls_wall_y[8*i+4:8*i+8] = pxls[33*i+14:33*i+18]
        puck_pxls_location[15*i:15*i+15] = pxls[33*i+18:33*i+33]

    while True:
        if setup.measure_extrinsics(imgs, puck_pxls_wall_x, puck_pxls_wall_y, puck_pxls_location):
                return 
        else:
            print("did not detect aruco")


def main():
    Get_calibration_data = False
    Solve_externals_and_zparams = False

    Normal_Mallet = False
    Print_Values = False
    Visual_Display = False

    #198.85  - 71.05 = 127.8 x
    #27.575

    if Solve_externals_and_zparams and not Get_calibration_data:
        getArucoPosSaved(3)
        return

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
        set_frame_rate(cam)
        configure_camera(cam)
        set_pixel_format(cam, "mono")

        if Get_calibration_data:
            get_calibration_data(cam, 4)
            cam_list.Clear()
            system.ReleaseInstance()
            sys.exit(1)
        
        get_external_matrix(cam)

        configure_gain(cam, gain_val=30.0)
        init_processes(cam, Normal_Mallet)
        cam.DeInit()
    except Exception as ex: #PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
    finally:
        cam.BeginAcquisition()
        time.sleep(0.01)
        cam.EndAcquisition()
        cam.DeInit()
        del cam
        cam_list.Clear()
        system.ReleaseInstance()

if __name__ == "__main__":
    setup = tracker.SetupCamera()
    main()

#(58.45+64.7)/2 = 61.575
#(29.45+35.7)/2 = 32.575
