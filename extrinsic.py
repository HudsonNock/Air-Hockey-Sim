import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import least_squares

def detect_aruco_markers(gray, x_offset):
    # Load the image
    
    # Load the predefined ArUco dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # Initialize the detector parameters
    parameters = cv2.aruco.DetectorParameters()
    
    # Detect markers
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None:
        top_right_corners = {}
        for i, marker_id in enumerate(ids.flatten()):
            # Extract top-right corner (third corner in OpenCV ArUco order)
            corners[i][0][1][0] += x_offset
            top_right = tuple(corners[i][0][1].astype(int))
            top_right_corners[marker_id] = top_right
        
        return top_right_corners
    
    return {}


def project_points(points_3d, rotation_matrix, tvec, intrinsic_matrix, dist_coeffs):
    """
    Project 3D points to 2D pixel coordinates with distortion.
    
    Args:
        points_3d: np.array of shape (N, 3) containing 3D points in world coordinates
        rvec: rotation vector (3,) or (3, 1)
        tvec: translation vector (3,) or (3, 1)
        intrinsic_matrix: 3x3 intrinsic camera matrix
        dist_coeffs: (k1, k2, p1, p2, k3) distortion coefficients
    
    Returns:
        points_2d: np.array of shape (N, 2) containing projected pixel coordinates
    """
    # Ensure points are float32
    points_3d = np.float32(points_3d)
    
    # Convert rotation vector to matrix
    R = rotation_matrix
    
    # Transform points to camera coordinates
    points_cam = np.dot(R, points_3d.T).T + tvec
    
    # Project to normalized image coordinates
    x = points_cam[:, 0] / points_cam[:, 2]
    y = points_cam[:, 1] / points_cam[:, 2]
    
    # Prepare for distortion
    r2 = x*x + y*y
    r4 = r2*r2
    r6 = r4*r2
    
    # Get distortion coefficients
    k1, k2, p1, p2, k3 = dist_coeffs
    
    # Apply radial distortion
    x_dist = x * (1 + k1*r2 + k2*r4 + k3*r6)
    y_dist = y * (1 + k1*r2 + k2*r4 + k3*r6)
    
    # Apply tangential distortion
    x_dist = x_dist + (2*p1*x*y + p2*(r2 + 2*x*x))
    y_dist = y_dist + (p1*(r2 + 2*y*y) + 2*p2*x*y)
    
    # Apply camera matrix
    fx, fy = intrinsic_matrix[0,0], intrinsic_matrix[1,1]
    cx, cy = intrinsic_matrix[0,2], intrinsic_matrix[1,2]
    skew = intrinsic_matrix[0,1]
    
    u = fx * x_dist + skew * y_dist + cx
    v = fy * y_dist + cy
    
    return np.column_stack((u, v))


def global_coordinate_zpixel(pixel_point, rot_mat, tvec, intrinsic_matrix, dist_coeffs, delta_z, z_pixel_map):
    # Extract camera matrix parameters
    fx = intrinsic_matrix[0,0]
    fy = intrinsic_matrix[1,1]
    cx = intrinsic_matrix[0,2]
    cy = intrinsic_matrix[1,2]
    skew = intrinsic_matrix[0,1]
    
    # Get rotation matrix
    # R = cv2.Rodrigues(rvec)[0]
    # R_inv = np.linalg.inv(R)
    R_inv = rot_mat.T
    
    # Normalize pixel coordinates
    u, v = pixel_point
    x_distorted = (u - cx - skew * (v - cy)/fy) / fx
    y_distorted = (v - cy) / fy
    
    # Iteratively solve for undistorted coordinates
    x = x_distorted
    y = y_distorted
    
    # Newton's method to remove distortion
    for _ in range(10):
        r2 = x*x + y*y
        r4 = r2*r2
        r6 = r4*r2
        
        k1, k2, p1, p2, k3 = dist_coeffs
        
        x_calc = x * (1 + k1*r2 + k2*r4 + k3*r6) + 2*p1*x*y + p2*(r2 + 2*x*x)
        y_calc = y * (1 + k1*r2 + k2*r4 + k3*r6) + p1*(r2 + 2*y*y) + 2*p2*x*y
        
        if abs(x_calc - x_distorted) < 1e-6 and abs(y_calc - y_distorted) < 1e-6:
            #print("Solved")
            break
            
        x = x - 1*(x_calc - x_distorted)
        y = y - 1*(y_calc - y_distorted)

    # Set up system of equations:
    # We know: point_world = R_inv @ (point_cam - tvec)
    # And point_cam = z_cam * [x, y, 1]
    # And point_world[2] = z_world

    # This means:
    # z_world = R_inv[2,0]*(z_cam*x - tvec[0]) + R_inv[2,1]*(z_cam*y - tvec[1]) + R_inv[2,2]*(z_cam - tvec[2])

    # Solve for z_cam:
    a = R_inv[2,0]*x + R_inv[2,1]*y + R_inv[2,2]
    b = -(R_inv[2,0]*tvec[0] + R_inv[2,1]*tvec[1] + R_inv[2,2]*tvec[2])

    z_cam = (delta_z + z_pixel_map[int(pixel_point[0] // 8), int(pixel_point[1]//8)] - b) / a

    # Now we can get the camera coordinates
    point_cam = z_cam * np.array([x, y, 1])
    
    # Convert to world coordinates
    point_world = R_inv @ (point_cam - tvec)
    
    return point_world

import numpy as np

def global_coordinate_vectorized_zpixel(pixel_points, rot_mat, tvec, intrinsic_matrix, dist_coeffs, delta_z, z_pixel_map, iterations=10, const_z=0):
    """
    Convert an array of pixel points (N x 2) into world coordinates (N x 3)
    given the camera parameters and distortion coefficients.

    Parameters:
      pixel_points : np.ndarray
          Array of pixel points with shape (N, 2).
      rot_mat : np.ndarray
          The 3x3 rotation matrix.
      tvec : np.ndarray
          The translation vector (3,).
      intrinsic_matrix : np.ndarray
          The camera intrinsic matrix.
      dist_coeffs : array_like
          The distortion coefficients (k1, k2, p1, p2, k3).
      z_world : float
          The desired world Z-coordinate (table height).
      iterations : int, optional
          Number of iterations for Newton's method (default is 10).

    Returns:
      point_world : np.ndarray
          Array of world coordinates for each input pixel point (shape (N, 3)).
    """
    # Extract camera matrix parameters.
    fx = intrinsic_matrix[0, 0]
    fy = intrinsic_matrix[1, 1]
    cx = intrinsic_matrix[0, 2]
    cy = intrinsic_matrix[1, 2]
    skew = intrinsic_matrix[0, 1]
    
    # Unpack pixel coordinates.
    u = pixel_points[:, 0]
    v = pixel_points[:, 1]
    
    # Compute normalized distorted coordinates.
    x_distorted = (u - cx - skew * (v - cy) / fy) / fx
    y_distorted = (v - cy) / fy
    
    # Initialize undistorted coordinates.
    x = x_distorted.copy()
    y = y_distorted.copy()
    
    # Newton's method for undistortion (vectorized for all points).
    k1, k2, p1, p2, k3 = dist_coeffs  # unpack distortion coefficients
    for _ in range(iterations):
        r2 = x**2 + y**2
        r4 = r2**2
        r6 = r2 * r4
        
        x_calc = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        y_calc = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) + p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        
        # Update x and y. (Here we run a fixed number of iterations.)
        x = x - (x_calc - x_distorted)
        y = y - (y_calc - y_distorted)
    
    # Compute the inverse rotation matrix.
    R_inv = rot_mat.T  # Since rot_mat is a rotation matrix, its inverse is its transpose.
    
    # Compute the coefficient 'a' for each point.
    # a = R_inv[2,0] * x + R_inv[2,1] * y + R_inv[2,2]
    a = R_inv[2, 0] * x + R_inv[2, 1] * y + R_inv[2, 2]
    
    # Compute the constant 'b' (same for all points).
    b = -(R_inv[2, 0] * tvec[0] + R_inv[2, 1] * tvec[1] + R_inv[2, 2] * tvec[2])
    
    if z_pixel_map is not None:
        indices = (pixel_points // 8).astype(int)
        z_world = delta_z + z_pixel_map[indices[:,0], indices[:,1]]
    else:
        z_world = np.full((pixel_points.shape[0]),const_z, dtype=np.float32)

    # Solve for z_cam for each point.
    z_cam = (z_world - b) / a  # Shape: (N,)
    
    # Compute camera coordinates for each point:
    # Each point_cam = z_cam * [x, y, 1]
    # Multiply each coordinate by the corresponding z_cam value.
    point_cam = np.stack([x * z_cam, y * z_cam, z_cam], axis=1)  # Shape: (N,3)
    
    # Convert to world coordinates:
    # Original: point_world = R_inv @ (point_cam - tvec)
    # For a set of points, we compute: (point_cam - tvec) @ R_inv.T
    point_world = (point_cam - tvec) @ R_inv.T
    
    return point_world

def z_height(z_param, p):
    return z_param[0] + z_param[1] * p[0] + z_param[2] * p[1] +\
            z_param[3] * p[0] * p[1] + z_param[4] * p[0]**2 + z_param[5] * p[1]**2 +\
            z_param[6] * p[0]**3 + z_param[7] * p[0]**2 * p[1] + z_param[8] * p[0] * p[1]**2 +\
            z_param[9] * p[1]**3

def z_height_vectorized(z_param, p):
    return z_param[0] + z_param[1] * p[:,0] + z_param[2] * p[:,1] +\
            z_param[3] * p[:,0] * p[:,1] + z_param[4] * p[:,0]**2 + z_param[5] * p[:,1]**2 +\
            z_param[6] * p[:,0]**3 + z_param[7] * p[:,0]**2 * p[:,1] + z_param[8] * p[:,0] * p[:,1]**2 +\
            z_param[9] * p[:,1]**3

def calibrate_extrinsic(img: np.ndarray, aruco_3d_points, intrinsic_matrix, dist_coeffs, x_offset):
    top_right_corners = detect_aruco_markers(img, x_offset)

    if len(top_right_corners) != 6:
        return None, None
    
    image_points = np.array([top_right_corners[i] for i in range(6)], dtype=np.float32)

    success, rvec, tvec = cv2.solvePnP(aruco_3d_points,
                                       image_points,
                                       intrinsic_matrix,
                                       dist_coeffs,
                                       flags=cv2.SOLVEPNP_ITERATIVE,
                                       useExtrinsicGuess=True,
                                       tvec=np.array([-0.54740303, -1.08125622, 2.45483598]),
                                       rvec=np.array([-2.4881045, -2.43093864, 0.81342852]))
    return rvec, tvec

def global_coordainte_vectorized_zworld_error(points, delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs, pxls):
    points = points.reshape(-1, 2)
    points_3d = np.hstack((points, np.expand_dims(delta_z + z_height_vectorized(z_params_world, points), axis=-1)))

    pixels = project_points(points_3d, rot_mat, tvec, intrinsic_matrix, dist_coeffs)
    # Compute the error in (x, y) only.
    return np.sum((pixels - pxls)**2, axis=-1)

def jacobian_vectorized(x, pxls, delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs):
    x = x.reshape(-1, 2)
    J = np.zeros_like(x)
    hx = np.array([1e-6,0])
    hy = np.array([0,1e-6])

    all_x = np.vstack((x+hx, x-hx, x+hy, x-hy))
    all_pxls = np.vstack((pxls, pxls, pxls, pxls))
    n = pxls.shape[0]

    # Compute gradients for all points at once
    errors = global_coordainte_vectorized_zworld_error(all_x, delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs, all_pxls)
    
    J[:, 0] = (errors[:n] - errors[n:2*n]) / (2e-6)  # Derivative w.r.t x0
    J[:, 1] = (errors[2*n:3*n] - errors[3*n:]) / (2e-6)  # Derivative w.r.t x1

    return J.ravel()

def global_coordinate_vectorized_zworld(pxls, rot_mat, tvec, intrinsic_matrix, dist_coeffs, delta_z, z_params_world):
    
    init_guess = global_coordinate_vectorized_zpixel(pxls, rot_mat, tvec, intrinsic_matrix, dist_coeffs, delta_z, None, const_z=z_params_world[0])[:,:2]

    result = minimize(lambda x: np.sum(global_coordainte_vectorized_zworld_error(x, delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs, pxls)),\
                       init_guess.ravel(),\
                          jac=lambda x: jacobian_vectorized(x, pxls, delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs),\
                              method='L-BFGS-B')
    #result = least_squares(global_coordainte_vectorized_zworld_error, init_guess.ravel(), method='trf', args=(delta_z, z_params_world, rot_mat, tvec, intrinsic_matrix,dist_coeffs,pxls))
    
    points = result.x.reshape(-1, 2)
    points_3d = np.hstack((points, np.expand_dims(delta_z + z_height_vectorized(z_params_world, points), axis=-1)))
    return points_3d

def Aruco_Ztable_Error(data, puck_pxls_location, aruco_pxls_all, intrinsic_matrix, dist_coeffs, puck_z, puck_locations):
    #wall_x: (x,,x-offset)
    #wall_y: (y, y-offset)
    #locations: corresponding to puck_locations
    aruco_3d = data[:18].reshape(6,3)
    z_params_world = data[18:]

    error = 0
    n = len(puck_pxls_location)

    for j in range(n):
        
        aruco_pxls = aruco_pxls_all[j*6:j*6+6]

        success, rvec, tvec = cv2.solvePnP(aruco_3d,
                                            aruco_pxls,
                                            intrinsic_matrix,
                                            dist_coeffs,
                                            flags=cv2.SOLVEPNP_ITERATIVE,
                                            useExtrinsicGuess=True,
                                            tvec=np.array([-0.54740303, -1.08125622, 2.45483598]),
                                            rvec=np.array([-2.4881045, -2.43093864, 0.81342852]))
        
        rot_mat = cv2.Rodrigues(rvec)[0]

        puck_pos = global_coordinate_vectorized_zworld(puck_pxls_location[j], rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params_world)[:,:2]
  
        error += np.sum(np.linalg.norm(puck_pos - puck_locations[j], axis=1)**2)

    print(error)
    return error

def solve_external_and_zparam(imgs, aruco_3d_guess, puck_pxls_location, intrinsic_matrix, dist_coeffs, puck_z, puck_locations, x_offset):
    
    n = len(puck_pxls_location)
    aruco_pxls = np.empty((6*n,2))
    for j in range(n):
        top_right_corners = detect_aruco_markers(imgs[j], x_offset)
        if len(top_right_corners) != 6:
            return None, None
        aruco_pxls[6*j:6*j+6] = np.array([top_right_corners[i] for i in range(6)], dtype=np.float32)

    initial_guess = aruco_3d_guess.flatten()
    init_z = np.zeros(shape=(10,), dtype=np.float32)
    init_z[0] = -(18.61)*10**(-3)
    initial_guess = np.concatenate((initial_guess, init_z))

    result = minimize(Aruco_Ztable_Error, initial_guess, method='Powell', args=(puck_pxls_location,aruco_pxls,intrinsic_matrix,dist_coeffs,puck_z,puck_locations))

    aruco_3d_points =  result.x[:18].reshape(6,3)
    z_params_world = result.x[18:]

    print(aruco_3d_points)
    print(z_params_world)
    print("Error")
    print(result.fun)
    #print(getError(result.x, pxls,image_points,intrinsic_matrix,puck_r,puck_z,height,width, puck_locations))
    return aruco_3d_points, z_params_world

def get_z_pixel_map(z_param_world, rot_mat, tvec, intrinsic_matrix, dist_coeffs):
    img_shape = (2048, 1536)
    z_pixel_map = np.full((img_shape[0] // 8, img_shape[1] // 8), np.nan, dtype=np.float32)

    x_vals = np.arange(0, img_shape[0], 8)
    y_vals = np.arange(0, img_shape[1], 8)
    xx, yy = np.meshgrid(x_vals, y_vals, indexing='ij')
    pxls = np.stack((xx.ravel(), yy.ravel()), axis=1)

    z_vals = global_coordinate_vectorized_zworld(pxls, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0, z_param_world)[:,2]  # Shape (N, 3)
    z_pixel_map[xx // 8, yy // 8] = z_vals.reshape(xx.shape)

    return z_pixel_map


"""
def getError(data, pxls, image_points, intrinsic_matrix, puck_r, puck_z, height, width, puck_locations):
    object_points = data[:18].reshape(6, 2)
    object_points = np.hstack((object_points, np.zeros((6, 1))))
    object_points[:,2] = 0
    z_params = data[18:]

    # Solve for rotation and translation (extrinsic parameters)
    dist_coeffs = np.array([-0.11734783, 0.11960238, 0.00017337, -0.00030401, -0.01158902], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points,
                                        image_points,
                                        intrinsic_matrix,
                                        dist_coeffs,
                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                        useExtrinsicGuess=True,
                                        tvec=np.array([-0.54740303, -1.08125622, 2.45483598]),
                                        rvec=np.array([-2.4881045, -2.43093864, 0.81342852]))
    
    rot_mat = cv2.Rodrigues(rvec)[0]

    error = 0
    
    #idx = 0
    for pixel in pxls[:5]:
        point_3D = global_coordinate(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params)[0:2]
        error += (point_3D[1] - puck_r)**2

    for pixel in pxls[5:9]:
        point_3D = global_coordinate(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params)[0:2]
        error += (point_3D[0] - puck_r)**2

    for pixel in pxls[9:14]:
        point_3D = global_coordinate(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params)[0:2]
        error += (point_3D[1] - (width - puck_r))**2

    for pixel in pxls[14:18]:
        point_3D = global_coordinate(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params)[0:2]
        error += (point_3D[0]- (height - puck_r))**2

    i = 0
    for pixel in pxls[18:]:
        point_3D = global_coordinate(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, puck_z, z_params)[0:2]
        error += np.sum((puck_locations[i] - point_3D)**2)
        i += 1

    return error

def get_measurments(img, intrinsic_matrix, object_guess, pxls, puck_z, puck_r, height, width, puck_locations):
    #pxls: x_axis, y_axis, x_axis_offset, y_axis_offset, centered
    #       5        5       4                4           16
    top_right_corners = detect_aruco_markers(img)

    if len(top_right_corners) != 6:
        return None, None, None, None
    
    image_points = np.array([top_right_corners[i] for i in range(6)], dtype=np.float32)

    initial_guess = object_guess.flatten()
    init_z = np.zeros(shape=(10,), dtype=np.float32)
    init_z[0] = -(18.61)*10**(-3)
    initial_guess = np.concatenate((initial_guess, init_z))

    result = minimize(getError, initial_guess, method='Powell', args=(pxls,image_points,intrinsic_matrix,puck_r,puck_z,height,width, puck_locations))

    object_points =  result.x[:18].reshape(6,3)
    z_params = result.x[18:]
    dist_coeffs = np.array([-0.11734783, 0.11960238, 0.00017337, -0.00030401, -0.01158902], dtype=np.float32)
    success, rvec, tvec = cv2.solvePnP(object_points,
                                        image_points,
                                        intrinsic_matrix,
                                        dist_coeffs,
                                        flags=cv2.SOLVEPNP_ITERATIVE,
                                        useExtrinsicGuess=True,
                                        tvec=np.array([-0.54740303, -1.08125622, 2.45483598]),
                                        rvec=np.array([-2.4881045, -2.43093864, 0.81342852]))
    
    rot_mat = cv2.Rodrigues(rvec)[0]
    print(object_points)
    print(z_params)
    print("Error")
    print(result.fun)
    print(getError(result.x, pxls,image_points,intrinsic_matrix,puck_r,puck_z,height,width, puck_locations))
    return object_points, rot_mat, tvec, z_params

def project_points_unknown_z(pts, rot_mat, tvec, intrinsic_matrix, dist_coeffs, z_params):
    #takes 3D points without Z
    # returns pxl corresponding to real world (x,y) using z_height

    def residual(pixel_flat):
        # reshape into (n,2) pixel coordinates
        pixels = pixel_flat.reshape(-1, 2)
        # Obtain the corresponding 3D points.
        # Note: global_coordinate_vectorized uses z_height_vectorized internally.
        points_3d = global_coordinate_vectorized(pixels, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0, z_params)
        # Compute the error in (x, y) only.
        return (points_3d[:, :2] - pts).ravel()
    
    initial_3d_guess = np.hstack((pts, np.full((pts.shape[0], 1), z_params[0])))
    pixel0 = project_points(initial_3d_guess, rot_mat, tvec, intrinsic_matrix, dist_coeffs)
    result = least_squares(residual, pixel0.ravel())
    pixels_solution = result.x.reshape(-1, 2)
    return pixels_solution


def z_error(z_params_global, z_params_pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, points):
    heights_global = z_height_vectorized(z_params_global, points)

    points_pxl = project_points_unknown_z(points, rot_mat, tvec, intrinsic_matrix, dist_coeffs, z_params_pixel)
    heights_pxl = z_height_vectorized(z_params_pixel, points_pxl)

    return np.sum(np.square(heights_global - heights_pxl))

def get_z_global(z_params, rot_mat, tvec, intrinsic_matrix, dist_coeffs, height, width):
    init_z = np.zeros(shape=(10,), dtype=np.float32)
    init_z[0] = -(18.61)*10**(-3)

    x_values = np.arange(0, height + 0.01, 0.01)
    y_values = np.arange(0, width + 0.01, 0.01)

    X, Y = np.meshgrid(x_values, y_values, indexing="ij")
    points = np.column_stack((X.ravel(), Y.ravel()))

    result = minimize(z_error, init_z, method="Powell", args=(z_params, rot_mat, tvec, intrinsic_matrix, dist_coeffs, points))
    return result.x
"""