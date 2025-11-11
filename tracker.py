import numpy as np
import cv2
import csv
import time
import os
from scipy.optimize import least_squares
from scipy.optimize import minimize
import agent_processing as ap
import json

import extrinsic

class CircularCameraBuffer():
    def __init__(self):
        self.array = np.empty((12*4,))
        self.get_arr = np.empty((5*4,))
        self.buffer_size = 12*4
        self.head = 0

    def put(self,arr):
        self.head = (self.head - 4) % self.buffer_size
        self.array[self.head:self.head+4] = arr

    def get(self):
        offsets = np.array([0, 4, 8, 20, 44])
        all_indices = (self.head + offsets[:, None] + np.arange(4)) % self.buffer_size
        self.get_arr[:] = np.take(self.array, all_indices).reshape(-1)
        return self.get_arr

class SetupCamera:
    def __init__(self):

        self.aruco_3d_points =  np.array([[-0.0775, 0.195 ,0],
                            [-0.07672, 0.815, 0],
                            [0.406, 1.08-0.011, 0],
                            [(207.8 - 8.1)/100, 0.7055, 0],
                            [(207.8 - 8.1)/100, 0.309, 0],
                            [0.3592, -0.00554, 0]], dtype=np.float32)
        
        self.intrinsic_matrix = np.array([[1.63911836e+03, 0.0, 1.01832305e+03],
                                [0.0, 1.63894064e+03, 7.72656464e+02],
                                [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.1278022,   0.15557611,  0.00117351, -0.00016628, -0.0654127 ])
        self.rotation_matrix = cv2.Rodrigues(np.array([-2.4881045, -2.43093864, 0.81342852]))[0]
        self.translation_vector = np.array([-0.54740303, -1.08125622,  2.45483598])

        self.puck_z = 3.75* 10**(-3)
        self.puck_r = 0.5 * 0.0636
        self.table_width = 0.992
        self.table_height = 1.993

        """
        self.puck_coords = np.array([[self.table_height - (0.161 + 0.0655/2), 0.1575 + 0.0655/2],
                                    [self.table_height - (0.1755 + 0.0655/2), 0.76 + 0.0655/2],
                                    [self.table_height - (0.2422 + 0.064/2), 0.466 + 0.064/2],
                                    [self.table_height - (0.473 + 0.064/2), 0.2197 + 0.064/2],
                                    [self.table_height - (0.467 + 0.064/2), self.table_width - (0.2095+0.064/2)], # 5
                                    [self.table_height - (0.7165 + 0.063/2), 0.466 + 0.063/2],
                                    [self.table_height - (0.8665+0.065/2), 0.221 + 0.065/2],
                                    [self.table_height - (0.8718 + 0.065/2), self.table_width - (0.187 + 0.065/2)],
                                    [0.9248 + 0.0254 + 0.065/2, 0.474 + 0.065/2],
                                    [0.7162 + 0.065/2, 0.208 + 0.065/2], # 10
                                    [0.768 + 0.065/2, self.table_width - (0.175 + 0.065/2)],
                                    [0.4987 + 0.0254 + 0.0648/2, 0.4555 + 0.0648/2],
                                    [0.309 + 0.065/2, 0.219 + 0.065/2],
                                    [0.359 + 0.065/2, self.table_width - (0.189 + 0.065/2)],
                                    [0.1325 + 0.0254 + 0.0655/2, 0.473 + 0.0655/2]]) #15
        """
        
        self.z_params = np.zeros(shape=(6,), dtype=np.float32)
        self.z_params[0] = -(18.61)*10**(-3)

        img_shape = (2048, 1536)
        self.z_pixel_map = np.full((img_shape[0] // 8, img_shape[1] // 8), self.z_params[0], dtype=np.float32)

        self.x_offset = 376

    def get_puck_pixel(self, frame):
        mask = cv2.inRange(frame, 150, 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        M = cv2.moments(contour)
        if len(contour) > 10 and M['m00'] != 0:
            img_pos = np.array([M['m10'] / M['m00'] + self.x_offset, M['m01'] / M['m00']])
            return img_pos
    
    def run_extrinsics(self, frame):
        self.aruco_3d_points = np.load("aruco_3d_points.npy")
        self.z_params_world = np.load("z_params.npy")
        rvec, tvec = extrinsic.calibrate_extrinsic(frame, self.aruco_3d_points, self.intrinsic_matrix, self.distortion_coeffs, self.x_offset)
        if rvec is not None and tvec is not None:
            self.rotation_matrix = cv2.Rodrigues(rvec)[0]
            self.translation_vector = tvec
            self.z_pixel_map = extrinsic.get_z_pixel_map(self.z_params_world, self.rotation_matrix, self.translation_vector, self.intrinsic_matrix, self.distortion_coeffs)
            print('Extrinsic calibration successful')
            return True

        else:
            print('Extrinsic calibration failed, waiting for next time')
            return False

    def solve_calibration(self, imgs, puck_pxls, puck_locations):
        aruco_3d_points_min, z_params = extrinsic.solve_external_and_zparam(imgs,\
                                                        self.aruco_3d_points,\
                                                        puck_pxls,\
                                                        self.intrinsic_matrix,\
                                                        self.distortion_coeffs,\
                                                        self.puck_z,\
                                                        puck_locations,\
                                                        self.x_offset)
        if aruco_3d_points_min is None:
            return False
        self.aruco_3d_points = aruco_3d_points_min
        self.z_params = z_params
        np.save("aruco_3d_points.npy", self.aruco_3d_points)
        np.save("z_params.npy", z_params)
        return True
    
    def see_aruco_pixels(self, img):
        top_right_corners = extrinsic.detect_aruco_markers(img)

        if len(top_right_corners) != 6:
            return False
        
        return True

class CameraTracker:
    def __init__(self, rotation_matrix, translation_vector, z_pixel_map, op_mallet_z):
        self.intrinsic_matrix = np.array([[1.63911836e+03, 0.0, 1.01832305e+03],
                                [0.0, 1.63894064e+03, 7.72656464e+02],
                                [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.1278022,   0.15557611,  0.00117351, -0.00016628, -0.0654127 ])

        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.z_pixel_map = z_pixel_map

        self.puck_z = 3.84* 10**(-3)
        self.puck_r = 0.5 * 0.0618

        self.op_mallet_z = op_mallet_z

        self.past_puck_pos = np.array([1.7, 0.5])
        self.past_op_mallet_pos = np.array([0.3, 0.5])
        self.past_past_puck_pos = np.array([1.7, 0.5])

        self.past_data = CircularCameraBuffer()
        self.x_offset = 376

    def track(self, frame):
        mask = cv2.inRange(frame, 150, 255)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None, None
        hierarchy = hierarchy[0]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        puck_pos = None
        mallet_pos = None

        for i, contour in enumerate(contours):
            if hierarchy[i][2] == -1 and len(contour) > 20:
                contour[:,0,0] += self.x_offset
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    img_pos = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
                    img_pos_int = tuple(map(int, img_pos))

                    if mask[img_pos_int[1], img_pos_int[0] - self.x_offset] == 0 and mallet_pos is None:
                        mallet_pos = extrinsic.global_coordinate_zpixel(img_pos,
                                self.rotation_matrix,
                                self.translation_vector,
                                self.intrinsic_matrix,
                                self.distortion_coeffs,
                                self.op_mallet_z,
                                self.z_pixel_map)[0:2]
                    elif mask[img_pos_int[1], img_pos_int[0] - self.x_offset] == 255 and puck_pos is None:
                        points_3D = extrinsic.global_coordinate_vectorized_zpixel(np.vstack((contour.squeeze(axis=1), img_pos)),\
                                self.rotation_matrix,\
                                self.translation_vector,\
                                self.intrinsic_matrix,\
                                self.distortion_coeffs,\
                                self.puck_z,
                                self.z_pixel_map)[:,:2]
                        puck_pos = self.fit_circle(points_3D[:-1], points_3D[-1:].squeeze())
                    
                    if puck_pos is not None and mallet_pos is not None:
                        break

        return puck_pos, mallet_pos
    
    def circle_fitting_score(self, center, points):
        """Compute the score based on distance from points to the circle."""
        distances = np.linalg.norm(points - center, axis=1)
        return np.sum(np.abs(distances - self.puck_r) < 0.0025)
    
    def fit_circle(self, points, centroid):        
        min_x = np.min(points[:,0])
        max_x = np.max(points[:,0])

        if max_x - min_x > 2*self.puck_r - 0.002:
            return centroid
        
        close_points = points[np.abs(points[:, 0] - min_x) <= 0.0012]
        min_xy = np.mean(close_points[:, 1])
        point_min_x = np.array([np.mean(close_points[:,0])+self.puck_r, min_xy])
        
        close_points = points[np.abs(points[:, 0] - max_x) <= 0.0012]
        max_xy = np.mean(close_points[:, 1])
        point_max_x = np.array([np.mean(close_points[:,0])-self.puck_r, max_xy])

        min_score = self.circle_fitting_score(point_min_x, points)
        max_score = self.circle_fitting_score(point_max_x, points)

        if min_score > 1.5 * max_score:
            return point_min_x
        elif max_score > 1.5 * min_score:
            return point_max_x
        
        if np.linalg.norm(point_min_x - self.past_puck_pos) < np.linalg.norm(point_max_x - self.past_puck_pos):
            return point_min_x
        return point_max_x
    
    def process_frame(self, frame):
        puck_pos, opponent_mallet_pos = self.track(frame)

        if opponent_mallet_pos is None:
            opponent_mallet_pos = self.past_op_mallet_pos
        
        if puck_pos is None:
            puck_pos = 2*self.past_puck_pos - self.past_past_puck_pos

        self.past_op_mallet_pos = opponent_mallet_pos
        self.past_past_puck_pos = self.past_puck_pos
        self.past_puck_pos = puck_pos

        self.past_data.put(np.concatenate([puck_pos, opponent_mallet_pos], axis=0))

        return puck_pos, opponent_mallet_pos

if __name__ == "__main__":
    track = SetupCamera()
    """
    img = cv2.imread("jump_bright.bmp", cv2.IMREAD_GRAYSCALE)[:,376:376+1296]
    #cv2.imshow("test", img)
    #cv2.waitKey(0)
    track.run_extrinsics(img)

    track = CameraTracker(track.rotation_matrix,
                          track.translation_vector,
                          track.z_pixel_map,
                          (120.94)*10**(-3))

    img = cv2.imread("jump.bmp", cv2.IMREAD_GRAYSCALE)[:,376:376+1296]
    while True:
        track.process_frame(img)
    """
    img1 = np.load('img_data_1.npy')
    img2 = np.load('img_data_2.npy')
    img3 = np.load('img_data_3.npy')
    imgs = [img1, img2, img3]

    pxls1 = np.load('pxls_data_1.npy')
    pxls2 = np.load('pxls_data_2.npy')
    pxls3 = np.load('pxls_data_3.npy')
    pxls = [pxls1, pxls2, pxls3]

    loc1 = np.load('location_data_1.npy')
    loc2 = np.load('location_data_2.npy')
    loc3 = np.load('location_data_3.npy')
    locs = [loc1, loc2, loc3]

    track.solve_calibration(imgs, pxls, locs)
