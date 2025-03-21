import numpy as np
import cv2
import csv
import time
import os
import sys
from scipy.optimize import least_squares
from scipy.optimize import minimize

# Add path for extrinsic module (adjust as needed)
sys.path.append(os.path.expanduser('~/air_hockey_ws/src/cv/cv'))
import extrinsic

class PuckTracker:
    def __init__(self):
        self.past_time = 0
        self.past_op_mallet_pos = np.array([0.3, 0.5])
        self.measured_past_op_mallet = False

        self.past_puck_pos = np.array([1.7, 0.5])
        self.past_puck_vel = np.array([0,0])
        self.measured_past_puck = False

        self.aruco_3d_points =  np.array([[-0.0775, 0.195 ,0],
                            [-0.07672, 0.815, 0],
                            [0.406, 1.08-0.011, 0],
                            [(207.8 - 8.1)/100, 0.7055, 0],
                            [(207.8 - 8.1)/100, 0.309, 0],
                            [0.3592, -0.00554, 0]], dtype=np.float32)
        
        
        # Camera parameters
        #self.intrinsic_matrix = np.array([[1.64122926e+03, 0.0, 1.01740071e+03],
        #                                  [0.0, 1.64159345e+03, 7.67420885e+02],
        #                                  [0.0, 0.0, 1.0]])
        self.intrinsic_matrix = np.array([[1.63911836e+03, 0.0, 1.01832305e+03],
                                [0.0, 1.63894064e+03, 7.72656464e+02],
                                [0.0, 0.0, 1.0]])
        #self.distortion_coeffs = np.array([-0.11734783, 0.11960238, 0.00017337, -0.00030401, -0.01158902])
        self.distortion_coeffs = np.array([-0.1278022,   0.15557611,  0.00117351, -0.00016628, -0.0654127 ])
        self.rotation_matrix = cv2.Rodrigues(np.array([-2.4881045, -2.43093864, 0.81342852]))[0]
        self.translation_vector = np.array([-0.54740303, -1.08125622,  2.45483598])

        self.puck_z = 3.84* 10**(-3)
        self.puck_r = 0.5 * 0.0618
        self.table_width = 0.9905
        self.table_height = 1.9885 
        self.op_mallet_z = 0
        self.op_mallet_r = 0.5 * 0.08

        self.mallet_r = 0.5*0.112

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

    def log_csv(self, position, dt):
        """
        Append tracking data to a CSV file.
        """
        csv_path = os.path.expanduser('~/air_hockey_ws/src/cv/cv/csv/position.csv')
        if not os.path.exists(csv_path):
            with open(csv_path, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'dt'])
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            if position[0] is not None:
                writer.writerow([position[0], position[1], dt])

    def log_data(self, mallet = False):
        """
        Print tracking data.
        """
        print("Puck")
        if self.past_puck_pos[0] is not None:
            print(f"  Position: x={self.past_puck_pos[0]}, y={self.past_puck_pos[1]}")
        if self.past_puck_vel[0] is not None:
            speed = np.sqrt(self.past_puck_vel[0]**2 + self.past_puck_vel[1]**2)
            print(f"  Velocity: x={self.past_puck_vel[0]}, y={self.past_puck_vel[1]}")
            print(f"  Speed: {speed}")

        if mallet:
            print("Opponent Mallet")
            if self.past_op_mallet_pos[0] is not None:
                print(f"  Position: x={self.past_op_mallet_pos[0]}, y={self.past_op_mallet_pos[1]}")

    def generate_top_down_view(self):
        """
        Create a top-down view of the table with the puck position.
        """
        
        top_down_image = np.ones((int(self.table_width * 500), int(self.table_height * 500), 3), dtype=np.uint8) * 255
        # Convert world coordinates to image coordinates.
        x_img = int(self.past_puck_pos[0] * 500)  # scale factor for x
        y_img = int((self.table_width - self.past_puck_pos[1]) * 500)  # invert y-axis for display
        cv2.circle(top_down_image, (x_img, y_img), int(self.puck_r * 500), (0, 255, 0), -1)

        x_img = int(self.past_op_mallet_pos[0] * 500)  # scale factor for x
        y_img = int((self.table_width - self.past_op_mallet_pos[1]) * 500)  # invert y-axis for display
        cv2.circle(top_down_image, (x_img, y_img), int(self.op_mallet_r * 500), (0, 255, 0), -1)
        cv2.imshow("top_down_table", top_down_image)
        cv2.waitKey(1)

    def set_mallet(self, mallet_z):
        self.op_mallet_z = mallet_z

    def set_time(self, time):
        self.past_time = time

    def get_puck_pixel(self, frame):
        mask = cv2.inRange(frame, 150, 255)

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        contour = contours[0]
        M = cv2.moments(contour)
        if len(contour) > 10 and M['m00'] != 0:
            img_pos = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
            return img_pos
    
    def track(self, frame):
        """
        Process a frame to track the specified target.
        Returns the target’s position in world coordinates
        """
        mask = cv2.inRange(frame, 150, 255)
        #cv2.imshow("masked", mask)
        #cv2.waitKey(1)
        # Find contours from the mask
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contours) == 0:
            return None, None
        hierarchy = hierarchy[0]

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        puck_pos = None
        mallet_pos = None
        for i, contour in enumerate(contours):
            if hierarchy[i][2] == -1 and len(contour) > 40:
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    img_pos = np.array([M['m10'] / M['m00'], M['m01'] / M['m00']])
                    img_pos_int = tuple(map(int, img_pos))

                    if mask[img_pos_int[1], img_pos_int[0]] == 0 and mallet_pos is None:
                        mallet_pos = extrinsic.global_coordinate_zpixel(img_pos,
                                self.rotation_matrix,
                                self.translation_vector,
                                self.intrinsic_matrix,
                                self.distortion_coeffs,
                                self.op_mallet_z,
                                self.z_pixel_map)[0:2]
                    elif mask[img_pos_int[1], img_pos_int[0]] == 255 and puck_pos is None:
                        points_3D = extrinsic.global_coordinate_vectorized_zpixel(np.vstack(contour.squeeze(axis=1), img_pos),\
                                self.rotation_matrix,\
                                self.translation_vector,\
                                self.intrinsic_matrix,\
                                self.distortion_coeffs,\
                                self.puck_z,
                                self.z_pixel_map)[:,:2]
                        #print("{" + ", ".join(f"({x}, {y})" for x, y in points_3D) + "}")
                        #print("A")
                        puck_pos = self.fill_circle(points_3D[:-1], points_3D[-1:]) #, centroid)
                        #print(puck_pos)
                        #center = extrinsic.project_points(np.array([np.append(puck_pos, self.puck_z)]), self.rotation_matrix, self.translation_vector, self.intrinsic_matrix, self.distortion_coeffs)
                        #mask_clr = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                        #mask_clr[tuple(map(int, center[0]))[1]][tuple(map(int,center[0]))[0]] = [0, 255, 0]
                        #cv2.imshow("masked", mask_clr[::-1, :])
                        #cv2.waitKey(1)
                    
                    if puck_pos is not None and mallet_pos is not None:
                        break

        return puck_pos, mallet_pos
    
    def circle_fitting_score(self, center, points):
        """Compute the score based on distance from points to the circle."""
        distances = np.linalg.norm(points - center, axis=1)  # Compute |P - C|
        return np.sum(np.abs(distances - self.puck_r) < 0.0025)
    
    def fill_circle(self, points, centroid):
        if len(points) < 10:
            return None  # Not enough points
        
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
        current_time = time.time()
        dt = current_time - self.past_time
        if dt <= 1e-8:
            dt = 1e-3
        self.past_time = current_time
        puck_pos, opponent_mallet_pos = self.track(frame)

        opponent_mallet_vel = np.array([0,0])
        puck_vel = np.array([0,0])

        if opponent_mallet_pos is None:
            opponent_mallet_pos = self.past_op_mallet_pos
            self.measured_past_op_mallet = False
        elif self.measured_past_op_mallet:
            opponent_mallet_vel = (opponent_mallet_pos - self.past_op_mallet_pos) / dt
            self.measured_past_op_mallet = True
            self.past_op_mallet_pos = opponent_mallet_pos
        else:
            self.measured_past_op_mallet = True
        
        if puck_pos is None:
            self.measured_past_puck = False
            puck_pos, puck_vel = self.puck_projection(dt)
            self.past_puck_pos = puck_pos
            self.past_puck_vel = puck_vel
        elif self.measured_past_puck:
            puck_vel = (puck_pos - self.past_puck_pos) / dt
            self.measured_past_puck = True
            self.past_puck_pos = puck_pos
            self.past_puck_vel = puck_vel
        else:
            #TODO Improve if this is insufficient
            #puck_vel = (puck_pos - self.past_puck_pos) / dt
            self.measured_past_puck = True
            self.past_puck_pos = puck_pos
            self.past_puck_vel = puck_vel

        return puck_pos, puck_vel, opponent_mallet_pos, opponent_mallet_vel
    
    def run_extrinsics(self, frame):
        rvec, tvec = extrinsic.calibrate_extrinsic(frame, self.aruco_3d_points, self.intrinsic_matrix, self.distortion_coeffs)
        if rvec is not None and tvec is not None:
            self.rotation_matrix = cv2.Rodrigues(rvec)[0]
            self.translation_vector = tvec
            print('Extrinsic calibration successful')
            return True

        else:
            print('Extrinsic calibration failed, waiting for next time')
            return False
        
    def load_extrinsics(self):
        self.aruco_3d_points = np.load("aruco_3d_points.npy")
        self.z_params_world = np.load("z_params.npy")
        self.z_pixel_map = extrinsic.get_z_pixel_map(self.z_params_world, self.rotation_matrix, self.translation_vector, self.intrinsic_matrix, self.distortion_coeffs)
        print("finished_loading")

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
        top_right_corners = extrinsic.detect_aruco_markers(img)

        if len(top_right_corners) != 6:
            return False
        
        return True
    
    def puck_projection(self, dt):
        #TODO Improve if this is insufficient
        puck_pos = self.past_puck_pos #+ self.past_puck_vel * dt
        return puck_pos, self.past_puck_vel

if __name__ == "__main__":
    track = PuckTracker()
    track.load_points()
    img = cv2.imread("jump_bright.bmp", cv2.IMREAD_GRAYSCALE)
    track.run_externals(img)

    img = cv2.imread("jump.bmp", cv2.IMREAD_GRAYSCALE)
    track.process_frame(img)
