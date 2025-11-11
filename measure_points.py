import extrinsic
import numpy as np
import cv2
from scipy.optimize import minimize

def getError(object_points):
    object_points = object_points.reshape(6, 2)
    object_points = np.hstack((object_points, np.zeros((6, 1))))

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

    error = 0# np.zeros((len(x_coord) + len(y_coord)+len(x_coord_offset)+len(y_coord_offset),))
    
    #idx = 0
    for pixel in x_coord:
        point_3D = extrinsic.reprojection(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0)
        error += (point_3D[1])**2
        #idx += 1

    for pixel in y_coord:
        point_3D = extrinsic.reprojection(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0)
        error += (point_3D[0])**2
        #idx += 1

    for pixel in x_coord_offset:
        point_3D = extrinsic.reprojection(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0)
        error += (point_3D[1] - 1.007)**2
        #idx += 1

    for pixel in y_coord_offset:
        point_3D = extrinsic.reprojection(pixel, rot_mat, tvec, intrinsic_matrix, dist_coeffs, 0)
        error += (point_3D[0]-1.99)**2
        #idx += 1

    return error

points = np.array([[-0.10243821,  0.10770151, -0.01291404],
 [-0.03992479,  0.91694174, -0.00734289],
 [ 0.36107301, 1.1130411,  -0.00209154],
 [ 1.94956585,  0.74220479, -0.0365173 ],
 [ 2.00409263,  0.32341144, 0.01461468],
 [ 0.34210291, -0.06308578, -0.01520587]])

z_params = np.array([-0.01271815, -0.00282956,  0.0086132,  -0.00245425,  0.00470319,  0.00500488])
np.save("aruco_3d_points.npy", points)
np.save("z_params.npy", z_params)

#img = cv2.imread("puck_far.bmp", cv2.IMREAD_GRAYSCALE)
#mask = cv2.inRange(img, 150, 255)
#cv2.imshow("masked", mask)
#cv2.waitKey(1)
while True:
    pass

#put image
img = cv2.imread("C:\\Users\\hudso\\Downloads\\table.bmp", cv2.IMREAD_COLOR)

red_mask = np.all(img == [0, 0, 255], axis=-1)  # Red pixels
green_mask = np.all(img == [0, 255, 0], axis=-1)  # Green pixels
blue_mask = np.all(img == [255, 0, 0], axis=-1)  # Blue pixels
yellow_mask = np.all(img == [0, 255, 255], axis=-1)  # Yellow pixels

# Get the coordinates of matching pixels
x_coord = np.column_stack(np.where(red_mask))[:, ::-1]  # Red
y_coord = np.column_stack(np.where(green_mask))[:, ::-1]  # Green
x_coord_offset = np.column_stack(np.where(blue_mask))[:, ::-1]  # Blue
y_coord_offset = np.column_stack(np.where(yellow_mask))[:, ::-1]  # Yellow

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

image_points, _ = extrinsic.detect_aruco_markers(img)

if len(image_points) != 6:
    print("did not detect")

image_points = np.array([image_points[i] for i in range(6)], dtype=np.float32)

"""
img_resized = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))

# Adjust points for the resized image
image_points_resized = image_points / 2  # Scale down points
x_coord_resized = x_coord / 2
y_coord_resized = y_coord / 2
x_coord_offset_resized = x_coord_offset / 2
y_coord_offset_resized = y_coord_offset / 2

# Draw circles around detected points
for point in y_coord_offset_resized:
    x, y = int(point[0]), int(point[1])
    cv2.circle(img_resized, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red filled circle

# Display the result
cv2.imshow("Downsized Image with Markers", img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
"""

# Given intrinsic matrix I (assumed known)
intrinsic_matrix = np.array([[1.64122926e+03, 0.00000000e+00, 1.01740071e+03],
                                [0.00000000e+00, 1.64159345e+03, 7.67420885e+02],
                                [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],
                                dtype=np.float32)

#initial guess
object_points_0 = np.array([[-0.0775, 0.195],
                            [-0.07672, 0.815],
                            [0.406, 1.08-0.011],
                            [(207.8 - 8.1)/100, 0.7055],
                            [(207.8 - 8.1)/100, 0.309],
                            [0.3592, -0.00554]], dtype=np.float32)

result = minimize(getError, object_points_0.flatten(), method='Powell')

#print(result.x.reshape(6, 2))
padded_array = np.hstack((result.x.reshape(6,2), np.zeros((6, 1))))

# Print in proper format
print("[")
print(",\n".join("    [" + ", ".join(f"{num:.6f}" for num in row) + "]" for row in padded_array))
print("]")
print(getError(object_points_0.flatten()))
print(getError(result.x))
