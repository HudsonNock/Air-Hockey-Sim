import numpy as np
import cv2
import csv
import time
import os

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

        self.puck_z = 3.6* 10**(-3)
        self.puck_r = 0.5 * 0.0629
        self.table_width = 0.992
        self.table_height = 1.993 

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
        if len(contour) > 30 and M['m00'] > 150:
            img_pos = np.array([M['m10'] / M['m00'] + self.x_offset, M['m01'] / M['m00']])
            return img_pos
        return None
    
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

    def measure_extrinsics(self, imgs, puck_pxls_wall_x, puck_pxls_wall_y, puck_pxls_location):
        aruco_3d_points_min, z_params = extrinsic.solve_external_and_zparam(imgs,\
                                                        self.aruco_3d_points,\
                                                        puck_pxls_wall_x,\
                                                        puck_pxls_wall_y,\
                                                        puck_pxls_location,\
                                                        self.intrinsic_matrix,\
                                                        self.distortion_coeffs,\
                                                        self.puck_r,\
                                                        self.puck_z,\
                                                        self.table_height,\
                                                        self.table_width,\
                                                        self.puck_coords)
        if aruco_3d_points_min is None:
            return False
        self.aruco_3d_points = aruco_3d_points_min
        self.z_params = z_params
        np.save("aruco_3d_points.npy", self.aruco_3d_points)
        np.save("z_params.npy", z_params)
        return True
    
    def see_aruco_pixels(self, img):
        top_right_corners = extrinsic.detect_aruco_markers(img, self.x_offset)

        if len(top_right_corners) != 6:
            return False
        
        return True

class CameraTracker:
    def __init__(self, rotation_matrix, translation_vector, z_pixel_map, op_mallet_z, cutoff):
        self.intrinsic_matrix = np.array([[1.63911836e+03, 0.0, 1.01832305e+03],
                                [0.0, 1.63894064e+03, 7.72656464e+02],
                                [0.0, 0.0, 1.0]])
        self.distortion_coeffs = np.array([-0.1278022,   0.15557611,  0.00117351, -0.00016628, -0.0654127 ])

        self.rotation_matrix = rotation_matrix
        self.translation_vector = translation_vector
        self.z_pixel_map = z_pixel_map

        self.puck_z = 3.6* 10**(-3)
        self.puck_r = 0.5 * 0.0629

        self.op_mallet_z = op_mallet_z
        self.op_mallet_r = 0.5 * 0.0997 # Guess here, measure later

        self.past_puck_pos = np.array([1.7, 0.5])
        self.past_op_mallet_pos = np.array([0.3, 0.5])
        self.past_past_puck_pos = np.array([1.7, 0.5])

        self.past_data = CircularCameraBuffer()
        self.x_offset = 376
        
        self.bounds = [1.993, 0.992] 
        self.thresh_map = np.minimum(np.tile(cutoff[:,None], (1,1296)), 225) + 25
        
        self.mask = np.empty((1536,1296), dtype=np.uint8)

    def track(self, frame):
        
        #cv2.inRange(frame, 230, 255, dst=self.mask)
        cv2.compare(frame, self.thresh_map, cv2.CMP_GE, dst=self.mask)
        #cv2.imshow("mask", self.mask[::2, ::2])
        #cv2.waitKey(1)

        contours, _ = cv2.findContours(self.mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        #contours_copy = [cnt.copy() for cnt in contours]
        
        contours = [
            cnt for cnt in contours
            if len(cnt) >= 35
            and cv2.contourArea(cnt) >= 100
        ]
        
        if len(contours) == 0:
            return None, None
        
        # Sort by y value of the first point
        #contours.sort(key=lambda c: c[0][0][1])
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        puck_pos = None
        mallet_pos = None
        
        puck_center = None
        puck_area = 0
        puck_contour = None
        joins = 0
        
        for contour in contours:
            contour[:,0,0] += self.x_offset
            M = cv2.moments(contour)
            img_pos = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
            img_pos_int = tuple(map(int, img_pos))

            if self.mask[img_pos_int[1], img_pos_int[0] - self.x_offset] == 0 and mallet_pos is None:
                mallet_pos = extrinsic.global_coordinate_zpixel(img_pos,
                        self.rotation_matrix,
                        self.translation_vector,
                        self.intrinsic_matrix,
                        self.distortion_coeffs,
                        self.op_mallet_z,
                        self.z_pixel_map)[0:2]
            elif self.mask[img_pos_int[1], img_pos_int[0] - self.x_offset] != 0:
                #if puck_center is not None:
                #    print(np.linalg.norm(puck_center - img_pos))
                if ((puck_center is None) or (np.linalg.norm(puck_center - img_pos) < 50)):
                    joins += 1
                    if puck_center is None:
                        puck_center = img_pos
                        puck_contour = contour.squeeze(axis=1)
                    else:
                        puck_center = (puck_area * puck_center + M['m00'] * img_pos) / (puck_area + M['m00'])
                        puck_area += M['m00']
                        puck_contour = np.vstack((puck_contour, contour.squeeze(axis=1)))
                        
        #if joins == 1:
        #    cv2.drawContours(self.mask, contours_copy, -1, 100, 2)
        #    cv2.imshow("contours", self.mask)
        #    cv2.waitKey(0)
                    
        if puck_contour is not None:
            puck_contour = np.vstack((puck_contour, puck_center))
            points_3D = extrinsic.global_coordinate_vectorized_zpixel(puck_contour,\
                        self.rotation_matrix,\
                        self.translation_vector,\
                        self.intrinsic_matrix,\
                        self.distortion_coeffs,\
                        self.puck_z,
                        self.z_pixel_map)[:,:2]
            puck_pos = self.fit_circle(points_3D[:-1], points_3D[-1:].squeeze(), joins)

        return puck_pos, mallet_pos
    
    def circle_fitting_score(self, centers, points):
        """Compute the score based on distance from points to the circle."""
        #(1,n,2) (m,1,2)
        distances = np.linalg.norm(points[None,:,:] - centers[:,None,:], axis=2)
        
        close = np.abs(distances - self.puck_r) < 0.002
        far = (distances - self.puck_r) > 0.002
        
        scores = close.sum(axis=1) - 2 * far.sum(axis=1)
        
        return scores
        #distances = np.linalg.norm(points - center, axis=1)
        #return np.sum(np.abs(distances - self.puck_r) < 0.002) - 2*np.sum(distances - self.puck_r > 0.002)
    
    def fit_circle(self, points, centroid, joins):    
        min_x = np.min(points[:,0])
        max_x = np.max(points[:,0])

        min_y = np.min(points[:,1])
        max_y = np.max(points[:,1])

        if max_x - min_x > 2*self.puck_r - 0.003 and max_y - min_y > 2*self.puck_r - 0.003 and joins == 1:
            return centroid
        
        close_points = points[np.abs(points[:, 0] - min_x) <= 0.0025]
        min_xy = np.mean(close_points[:, 1])
        point_min_x = np.array([np.mean(close_points[:,0])+self.puck_r, min_xy])
        
        close_points = points[np.abs(points[:, 0] - max_x) <= 0.0025]
        max_xy = np.mean(close_points[:, 1])
        point_max_x = np.array([np.mean(close_points[:,0])-self.puck_r, max_xy])
        
        close_points = points[np.abs(points[:,1] - min_y) <= 0.0025]
        min_yx = np.mean(close_points[:,0])
        point_min_y = np.array([min_yx, np.mean(close_points[:,1])+self.puck_r])
        
        close_points = points[np.abs(points[:,1] - max_y) <= 0.0025]
        max_yx = np.mean(close_points[:,0])
        point_max_y = np.array([max_yx, np.mean(close_points[:,1])-self.puck_r])
        
        right_col_width = 0.005
        if centroid[0] < 1.6:
            right_col_width = 0.003
        right_col = points[np.abs(points[:,0] - max_x) <= right_col_width]
        
        r_width = np.max(right_col[:,1]) - np.min(right_col[:,1])

        r_width = max_y - min_y
        
        score_points = np.array([centroid, point_min_x, point_min_y, point_max_x, point_max_y, np.zeros((2,))])
        
        if r_width < 2*self.puck_r:
            if centroid[0] > 1.6:
                y_min = np.array([point_max_x[0]+self.puck_r, right_col[np.argmin(right_col[:,1])][1]])
                y_max = np.array([point_max_x[0]+self.puck_r, right_col[np.argmax(right_col[:,1])][1]])
            else:
                y_min = right_col[np.argmin(right_col[:,1])]
                y_max = right_col[np.argmax(right_col[:,1])]
            offset = np.sqrt(self.puck_r**2 - (r_width/2)**2)
            y_contendor = (y_min + y_max) / 2
            y_contendor[0] += offset
            
            score_points[5] = y_contendor
        
        scores = self.circle_fitting_score(score_points, points)

        #desmos_str = "[" + ", ".join(f"({x}, {y})" for x, y in points) + "]"
        #print("--")
        #print(desmos_str)
        #print(scores)
        #print("A3")
        #print(scores)
        
        sorted_indices = np.argsort(scores)
        #print(score_points[sorted_indices[-1]])

        return score_points[sorted_indices[-1]]
    
    def process_frame(self, frame, top_down_view=False, printing=False):
        puck_pos, opponent_mallet_pos = self.track(frame)
        if opponent_mallet_pos is None:
            opponent_mallet_pos = self.past_op_mallet_pos
        
        if puck_pos is None:
            puck_pos = self.past_puck_pos #2*self.past_puck_pos - self.past_past_puck_pos

        self.past_op_mallet_pos = opponent_mallet_pos
        self.past_past_puck_pos = self.past_puck_pos
        self.past_puck_pos = puck_pos

        self.past_data.put(np.concatenate([self.bounds - puck_pos, self.bounds - opponent_mallet_pos], axis=0))
        
        if printing:
            print("---")
            print(np.array(self.bounds) - puck_pos)
            print(opponent_mallet_pos)
        
        if top_down_view:
        	self.generate_top_down_view(puck_pos, opponent_mallet_pos)
        	
    def generate_top_down_view(self, puck_pos, op_mallet_pos):

        top_down_image = np.ones((int(self.bounds[1] * 500), int(self.bounds[0] * 500), 3), dtype=np.uint8) * 255
        # Convert world coordinates to image coordinates.
        x_img = int((self.bounds[0] - puck_pos[0]) * 500)  # scale factor for x
        y_img = int(puck_pos[1] * 500)  # invert y-axis for display
        cv2.circle(top_down_image, (x_img, y_img), int(self.puck_r * 500), (0, 255, 0), -1)

        x_img = int((self.bounds[0] - op_mallet_pos[0]) * 500)  # scale factor for x
        y_img = int(op_mallet_pos[1] * 500)  # invert y-axis for display
        cv2.circle(top_down_image, (x_img, y_img), int(self.op_mallet_r * 500), (255, 255, 0), -1)

        cv2.imshow("top_down_table", top_down_image)
        cv2.waitKey(1)

if __name__ == "__main__":
    track = SetupCamera()
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

