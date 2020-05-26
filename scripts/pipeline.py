import glob
import os
import pickle

from scripts.color_thresholder import hsl_select
from scripts.combined_gradient_filter import *


def gradient_threshold(image, params):
    gradx = abs_sobel_thresh(image,
                             orient='x',
                             sobel_kernel=params['k_size_x'],
                             thresh=(params['min_x'], params['max_x'])
                             )
    grady = abs_sobel_thresh(image,
                             orient='y',
                             sobel_kernel=params['k_size_y'],
                             thresh=(params['min_y'], params['max_y'])
                             )
    mag_binary = mag_thresh(image,
                            sobel_kernel=params['k_size_mag'],
                            mag_thresh=(params['min_mag'], params['max_mag'])
                            )
    dir_binary = dir_threshold(image,
                               sobel_kernel=params['k_size_dir'],
                               thresh=(params['min_dir'], params['max_dir'])
                               )
    combined_binary = combine(gradx, grady, mag_binary, dir_binary)
    return combined_binary


def color_threshold(image, params):
    hsl_mask = hsl_select(image, params['hsl_min'], params['hsl_max'])
    return hsl_mask


def calibrate_camera(params):
    grid_size = params['chessboard_size']

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane.

    # Make a list of calibration images
    images = glob.glob(f'{params["image_calib_folder"]}/calibration*.jpg')

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, grid_size, corners, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)

    cv2.destroyAllWindows()

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    # Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
    dist_pickle = {"mtx": mtx, "dist": dist}
    pickle.dump(dist_pickle, open(params['undistort'], "wb"))

    return mtx, dist


def undistort(image, params):
    if os.path.exists(params['undistort']):
        coefficients = pickle.load(open(params['undistort'], 'rb'))
        mtx = coefficients['mtx']
        dist = coefficients['dist']
    else:
        mtx, dist = calibrate_camera(params)

    return cv2.undistort(image, mtx, dist, None, mtx)


def find_perspective_lines(image):
    height, width = image.shape

    ground_offset = 35
    x_bottom_width = 835
    x_top_width = 115
    y_height = height * 5 / 8

    vertices = np.array([[width // 2 - x_bottom_width // 2, height - ground_offset],
                         [width // 2 - x_top_width // 2, y_height],
                         [width // 2 + x_top_width // 2, y_height],
                         [width // 2 + x_bottom_width // 2, height - ground_offset]], dtype=np.float32)

    output = cv2.polylines(convert_to_image(np.dstack((image, image, image))), [vertices.astype(np.int32)], True,
                           (0, 255, 255), 4)

    dist = np.array([[width // 2 - x_bottom_width // 2, height],
                     [width // 2 - x_bottom_width // 2, 0],
                     [width // 2 + x_bottom_width // 2, 0],
                     [width // 2 + x_bottom_width // 2, height]], dtype=np.float32)
    output = cv2.polylines(output, [dist.astype(np.int32)], True, (255, 0, 255), 4)

    M = cv2.getPerspectiveTransform(vertices, dist)
    perspective = cv2.warpPerspective(convert_to_image(image), M, (width, height), flags=cv2.INTER_LINEAR)

    cv2.imshow("Outlines", output)
    cv2.imshow("Perspective", perspective)

    return perspective


def image_pipeline(image, params):
    undistorted = undistort(image, params)

    gradients = gradient_threshold(undistorted, params)
    colors = color_threshold(undistorted, params)

    cv2.imshow('undistorted', undistorted)
    cv2.imshow('grads', convert_to_image(gradients))
    cv2.imshow('colors', convert_to_image(colors))

    combined_threshold = np.logical_or(gradients, colors)

    find_perspective_lines(combined_threshold)

    output = combined_threshold
    return output
