import glob
import pickle

import cv2
import numpy as np
import matplotlib.image as mpimg

# Code extracted from udacity
# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from Put your deep learning skills to the test with this project! Train a deep neural network to drive a car like you!
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

# Read the data
calibration_images_filenames = glob.glob('camera_cal/calibration*.jpg')
test_images_filenames = glob.glob('test_images/*.jpg')

calibration_images = [mpimg.imread(f) for f in calibration_images_filenames]
test_images = [mpimg.imread(f) for f in test_images_filenames]

# Helper functions
def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
def get_calibration_corners(calibration_images, nx=9, ny=6):
    """
    See https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    Args:
        calibration_images (list): list of matplotlib images with the chessboard
        nx (int): number of internal corners in the x axis
        ny (int): number of internal corners in the y axis
    Returns:
        list, list: 3d points in real world space, 2d points in image plane
    """
    objp = np.zeros((ny*nx, 3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Step through the list and search for chessboard corners
    for image in calibration_images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = to_grayscale(image)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    return objpoints, imgpoints


def calibrate_camera(calibration_images):
    """
    Calibrate the camera using a set of chessboard images
    Returns:
       camera_matrix and distortion_coefficient
    """
    _img = calibration_images[0]
    img_size = (_img.shape[1], _img.shape[0])

    objpoints, imgpoints = get_calibration_corners(calibration_images)
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                       img_size, None, None)
    return mtx, dist

def _save_calibration_result():
    mtx, dist = calibrate_camera(calibration_images)
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open("calibration_data/wide_dist_pickle.p", "wb"))

def _load_calibration_result():
    p = pickle.load(open("calibration_data/wide_dist_pickle.p", "rb"))
    return p['mtx'], p['dist']

# Apply a distortion correction to raw images.
# To do this we need the data we calculate in the previous section.
# _save_calibration_result()
camera_matrix, distortion_coeff = _load_calibration_result()

def undistort(image, camera_matrix, distortion_coeff):
    """Return the image undistorted"""
    return cv2.undistort(image, camera_matrix,
                         distortion_coeff, None, camera_matrix)

def _test_undistort():
    """Do this in all the images"""
    img = cv2.imread('test_images/test3.jpg')
    dst = undistort(img, camera_matrix, distortion_coeff)
    # cv2.imwrite('calibration_test_un.jpg', dst)
    diff = cv2.subtract(img, dst)
    cv2.imwrite('diff.jpg', diff)

_test_undistort()

# Use color transforms, gradients, etc., to create a thresholded binary image.

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_thresh(img, sobel_kernel=3, thresh=(0, 255)):
    # Define a function to return the magnitude of the gradient
    # for a given sobel kernel size and threshold values
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Define a function to threshold an image for a given range and Sobel kernel
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output


# Apply a perspective transform to rectify binary image ("birds-eye view").
def flatten_bird_eye(image):
    pass

# Detect lane pixels and fit to find the lane boundary.
def detect_lane_pixesl(image):
    pass

def fit_to_pixels(points):
    pass

# Determine the curvature of the lane and vehicle position with respect to center.
def get_polynomial(points):
    pass

# Warp the detected lane boundaries back onto the original image.
def draw_lanes(lanes):
    # extracted from udacity
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    plt.imshow(result)
