#%%
import numpy as np
import cv2
import glob
#%%
x2pix = 3.7/660
y2pix =  3/72

def get_camera_calibation():
    calibration_images_dir = "./camera_cal/"
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    images = glob.glob(calibration_images_dir+"*.jpg")
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
        gray.shape
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    return mtx, dist

def undistort(img, objpoints, imgpoints):
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    img_size = (img.shape[1], img.shape[0])
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    return mtx, dst

def get_prespective(img, unwrapped = False):
    h, w = 720, 1280
    src = np.float32([(525,464),
                      (630,464),
                      (300,682),
                      (700,682)])

    dst = np.float32([(460,0),
                      (w-476,0),
                      (460,h),
                      (w-460,h)])

    if(unwrapped):
        src , dst = dst, src
    M = cv2.getPerspectiveTransform(src, dst)
    return M

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return binary_output

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def hsv_select(img, thresh=(0, 255)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:,:,2]
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel > thresh[0]) & (v_channel <= thresh[1])] = 1
    return binary_output

def combined_thresholds(image):
    gradx = abs_sobel_thresh(image, orient='x', thresh=(13, 120))
    grady = abs_sobel_thresh(image, orient='y', thresh=(26, 100))
    hls_thresh = hls_select(image,  thresh=(101, 255))
    hsv_tresh = hsv_select(image,  thresh=(51, 255))
    mag_thresh = mag_threshold(image, mag_thresh=(31, 100))
    dir_thresh = dir_threshold(image, sobel_kernel=15, thresh=(.75, 1.25))
    combined_binary = np.zeros_like(dir_thresh)
    combined_binary[((gradx == 1) & (grady == 1)) | ((mag_thresh == 1) & (dir_thresh == 1)) | ((hls_thresh == 1) & (hsv_tresh == 1))] = 1
    return combined_binary


def lane_curvature(left_fit, right_fit):
    y_eval = 600
    y1 = (2 * left_fit[0] * y_eval + left_fit[1]) * x2pix / y2pix
    y2 = 2 * left_fit[0] * x2pix / (y2pix * y2pix)
    curvature = ((1 + y1 * y1) ** (1.5)) / np.absolute(y2)
    return curvature

def dist_from_center(frame, left_fit, right_fit):
    car_pos = frame.shape[1] / 2
    lane_center = (left_fit[-1] + right_fit[-1]) // 2
    distance = (car_pos - lane_center) * x2pix
    return distance