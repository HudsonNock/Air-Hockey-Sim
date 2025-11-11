import numpy as np
import cv2
import glob
import os
import matplotlib.pyplot as plt

def calibrate_camera(image_dir='./', image_format='img*.jpg', board_size=(21, 11), square_size=1.0, debug=True):
    """
    Calibrate a camera using checkerboard images with improved detection.
    
    Parameters:
    - image_dir: Directory containing the checkerboard images
    - image_format: Format of the images (e.g., 'img*.jpg')
    - board_size: Size of the checkerboard (inner corners)
    - square_size: Size of the checkerboard squares (arbitrary unit is fine)
    - debug: If True, saves debug images showing corner detection
    
    Returns:
    - ret: RMS re-projection error
    - mtx: Camera matrix (intrinsics)
    - dist: Distortion coefficients
    - rvecs: Rotation vectors for each image
    - tvecs: Translation vectors for each image
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2) * square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of images
    images = glob.glob(os.path.join(image_dir, image_format))
    
    if not images:
        print(f"No images found in {image_dir} with format {image_format}")
        return None, None, None, None, None
    
    print(f"Found {len(images)} images. Processing...")
    
    # Create debug directory if it doesn't exist
    if debug and not os.path.exists('debug_images'):
        os.makedirs('debug_images')
    
    # Criteria for corner detection
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            print(f"Failed to load image: {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Try different preprocessing techniques to improve corner detection
        processed_gray = gray.copy()
        
        # Adaptive histogram equalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_gray = clahe.apply(processed_gray)
        
        # Try to find the chessboard corners with multiple algorithms and flags
        detection_methods = [
            # Standard method
            {
                "method": "Standard", 
                "flag": cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            },
            # Add filter for fast check
            {
                "method": "Fast", 
                "flag": cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
            },
            # Try with extra filtering
            {
                "method": "Filtered", 
                "flag": cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FILTER_QUADS
            },
            # Try exhaustive search
            {
                "method": "Exhaustive", 
                "flag": cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_EXHAUSTIVE
            }
        ]
        
        # Try to find chessboard with different methods
        corners = None
        successful_method = None
        
        for method_info in detection_methods:
            method_name = method_info["method"]
            flag = method_info["flag"]
            
            print(f"Trying {method_name} method on {fname}...")
            ret, detected_corners = cv2.findChessboardCorners(processed_gray, board_size, flag)
            
            if ret:
                print(f"Success with {method_name} method on {fname}")
                corners = detected_corners
                successful_method = method_name
                break
        
        # If we found corners with any method
        if corners is not None:
            objpoints.append(objp)
            
            # Refine the corners
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            
            # Draw and display the corners (for debugging)
            if debug:
                img_copy = img.copy()
                cv2.drawChessboardCorners(img_copy, board_size, refined_corners, True)
                
                # Save debug image
                debug_filename = f"debug_images/detected_{os.path.basename(fname)}"
                
                # Add method name to filename
                base, ext = os.path.splitext(debug_filename)
                debug_filename = f"{base}_{successful_method}{ext}"
                
                cv2.imwrite(debug_filename, img_copy)
                print(f"Debug image saved as {debug_filename}")
                
                # If this is the first successful detection, show it for immediate feedback
                if len(objpoints) == 1:
                    plt.figure(figsize=(12, 8))
                    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                    plt.title('Original Image')
                    plt.subplot(122), plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
                    plt.title('Detected Corners')
                    plt.tight_layout()
                    plt.savefig(f"debug_images/first_detection_comparison.png")
                    print("First successful detection visualization saved")
        else:
            print(f"Could not find checkerboard corners in {fname} with any method")
            
            # Save problematic images for inspection
            if debug:
                problem_dir = 'debug_images/problem_images'
                if not os.path.exists(problem_dir):
                    os.makedirs(problem_dir)
                cv2.imwrite(f"{problem_dir}/failed_{os.path.basename(fname)}", img)
    
    if not objpoints:
        print("No valid checkerboard patterns found in any images")
        return None, None, None, None, None
        
    print(f"Successfully found corners in {len(objpoints)} images out of {len(images)}")
    
    if len(objpoints) < 3:
        print("Warning: At least 3 good images are recommended for reliable calibration")
        if len(objpoints) == 0:
            return None, None, None, None, None
    
    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Calculate re-projection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    print(f"Total error: {mean_error/len(objpoints)}")
    
    return ret, mtx, dist, rvecs, tvecs

def verify_board_dimensions(image_path, board_size=(21, 11)):
    """
    Tool to help verify if the board dimensions are correct.
    Displays the image with gridlines to help count corners.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image {image_path}")
        return
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Display grid lines for easier counting
    h, w = gray.shape
    img_copy = img.copy()
    
    # Draw vertical and horizontal grid lines
    cols = board_size[0] + 1
    rows = board_size[1] + 1
    
    # Estimate grid spacing
    grid_w = w // cols
    grid_h = h // rows
    
    # Draw grid
    for i in range(cols+1):
        x = i * grid_w
        cv2.line(img_copy, (x, 0), (x, h), (0, 255, 0), 1)
        if i < cols:
            cv2.putText(img_copy, str(i), (x+5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    for i in range(rows+1):
        y = i * grid_h
        cv2.line(img_copy, (0, y), (w, y), (0, 255, 0), 1)
        if i < rows:
            cv2.putText(img_copy, str(i), (5, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Create output directory
    if not os.path.exists('debug_images'):
        os.makedirs('debug_images')
        
    # Save the grid image for reference
    grid_filename = f"debug_images/grid_check_{os.path.basename(image_path)}"
    cv2.imwrite(grid_filename, img_copy)
    print(f"Grid check image saved as {grid_filename}")

if __name__ == "__main__":
    # Specify your image directory and format
    image_dir = "C:\\Users\\hudso\\Downloads\\checkerboard"  # Current directory
    image_format = 'img*.bmp'  # Adjust as needed for your image format
    
    # First image for verification
    first_image = glob.glob(os.path.join(image_dir, image_format))
    if first_image:
        print("First, let's verify if your board size is correct...")
        verify_board_dimensions(first_image[0], board_size=(20, 9))
        print("\nCheck the generated grid image in the debug_images folder.")
        print("If the grid doesn't align with your checkerboard, adjust the board_size parameter.")
        print("\nPress Enter to continue with calibration, or Ctrl+C to exit and adjust parameters.")
        input()
    
    # Define the checkerboard dimensions (inner corners)
    board_size = (20, 9)  # Adjust as needed
    
    # Run calibration with debug enabled
    ret, mtx, dist, rvecs, tvecs = calibrate_camera(
        image_dir=image_dir,
        image_format=image_format,
        board_size=board_size,
        square_size=1.0,  # Arbitrary unit is fine for intrinsics
        debug=True
    )
    
    if mtx is not None:
        # Print results
        print("\nCamera Matrix (Intrinsics):")
        print(mtx)
        print("\nDistortion Coefficients:")
        print(dist)
        
        # Save calibration results
        np.savez('calibration_results.npz', mtx=mtx, dist=dist)
        print("Calibration results saved to calibration_results.npz")
        
        # Create undistorted version of the first image
        if first_image:
            sample_img = cv2.imread(first_image[0])
            if sample_img is not None:
                h, w = sample_img.shape[:2]
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
                
                # Undistort
                dst = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)
                
                # Crop the image
                x, y, w, h = roi
                dst = dst[y:y+h, x:x+w]
                
                # Save sample undistorted image
                cv2.imwrite('undistorted_sample.jpg', dst)
                print("Sample undistorted image saved as 'undistorted_sample.jpg'")
    else:
        print("Calibration failed. Please check the debug images and adjust parameters as needed.")