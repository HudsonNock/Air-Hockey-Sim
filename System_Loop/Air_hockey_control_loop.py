#!/usr/bin/env python3
import PySpin
import cv2
import numpy as np
import sys
import os
import tracker
import time

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
    track.load_extrinsics()
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

            if track.run_extrinsics(image_data):
                break
    except KeyboardInterrupt:
        print("Image acquisition interrupted by user.")
    finally:
        cam.EndAcquisition()
        print("External ended.")

def acquire_images(cam, print_values, visual):
    # Begin image acquisition
    cam.BeginAcquisition()
    print("Acquisition started...")
    track.set_time(time.time())
    
    try:
        while True:
            # Retrieve next image with a timeout of 1000ms
            image_result = cam.GetNextImage()

            image_data = image_result.GetNDArray()

            #TODO get mallet pos from motor encoders

            puck_pos, puck_vel, op_mallet_pos, op_mallet_vel = track.process_frame(image_data)

            

            if print_values:
                track.log_data()

            if visual:
                track.generate_top_down_view()
            
            image_result.Release()

    except KeyboardInterrupt:
        print("Image acquisition interrupted by user.")
    finally:
        cam.EndAcquisition()
        cv2.destroyAllWindows()
        print("Acquisition ended.")

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
        pxl = track.get_puck_pixel(image_data)
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
        np.save(f"img_data_{j}.npy", image_data)
        if track.see_aruco_pixels(image_data):
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
        if track.measure_extrinsics(imgs, puck_pxls_wall_x, puck_pxls_wall_y, puck_pxls_location):
                return 
        else:
            print("did not detect aruco")


def main():
    Get_calibration_data = False
    Tune_On_Saved_Data = True

    Normal_Mallet = True
    Print_Values = True
    Visual_Display = False
#198.85  - 71.05 = 127.8 x
#27.575
    if Normal_Mallet:
        track.set_mallet((120.94)*10**(-3))
    else:
        track.set_mallet(0)

    if Tune_On_Saved_Data:
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

        if Get_calibration_data:
            get_calibration_data(cam, 2)
            cam_list.Clear()
            system.ReleaseInstance()
            sys.exit(1)
        
        get_external_matrix(cam)

        configure_gain(cam, gain_val=30.0)
        acquire_images(cam, Print_Values, Visual_Display)
        cam.DeInit()
    except PySpin.SpinnakerException as ex:
        print("Error: %s" % ex)
    finally:
        del cam
        cam_list.Clear()
        system.ReleaseInstance()

if __name__ == "__main__":
    track = tracker.PuckTracker()
    main()

#(58.45+64.7)/2 = 61.575
#(29.45+35.7)/2 = 32.575