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


def find_perspective_lines(image, params, show_plots=False):
    height, width = image.shape

    if 'M' not in params or 'M_inv' not in params:
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

        if show_plots:
            cv2.imshow("Outline", output)

        M = cv2.getPerspectiveTransform(vertices, dist)
        M_inv = cv2.getPerspectiveTransform(dist, vertices)

        params['M'] = M
        params['M_inv'] = M_inv
    else:
        M = params['M']
        M_inv = params['M_inv']

    perspective = cv2.warpPerspective(convert_to_image(image), M, (width, height), flags=cv2.INTER_LINEAR)

    return perspective, M, M_inv


def hist(img):
    histogram = np.sum(img[img.shape[0] // 2:, :], axis=0)

    return histogram


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0] // nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this

        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    try:
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1 * ploty ** 2 + 1 * ploty
        right_fitx = 1 * ploty ** 2 + 1 * ploty

    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    right_points = np.vstack((right_fitx, ploty)).transpose()
    left_points = np.vstack((left_fitx, ploty)).transpose()

    cv2.polylines(out_img, [left_points.astype(np.int)], False, (0, 255, 255))
    cv2.polylines(out_img, [right_points.astype(np.int)], False, (0, 255, 255))

    return out_img, left_points, right_points, left_fit, right_fit


def draw_lines(image, left_points, right_points, M_inv, alpha=.25):
    left_points = left_points.reshape((-1, 1, 2))
    right_points = right_points.reshape((-1, 1, 2))

    left_points = cv2.perspectiveTransform(left_points, M_inv)
    right_points = cv2.perspectiveTransform(right_points, M_inv)

    image = np.copy(image)
    overlay = np.copy(image)

    cv2.fillPoly(overlay, [
        np.array([left_points[0][0], left_points[-1][0], right_points[-1][0], right_points[0][0]], dtype=np.int)],
                 (0, 255, 0))

    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    cv2.polylines(image, [left_points.reshape((-1, 2)).astype(np.int)], False, (0, 255, 255), thickness=5)
    cv2.polylines(image, [right_points.reshape((-1, 2)).astype(np.int)], False, (0, 255, 255), thickness=5)

    return image


def calculate_curvature(fit, y, params):
    mx, my = params['xm_per_pix'], params['ym_per_pix']
    y = y * my

    a_mul = mx / (my ** 2)
    b_mul = mx / my

    return (1 + (2 * fit[0] * a_mul * y + b_mul * fit[1]) ** 2) ** 1.5 / (np.abs(a_mul * 2 * fit[0]))


def print_curvature(image, left_curvature, right_curvature):
    text = f"Left Curve: {int(round(left_curvature, 0)):4d}m Right Curve: {int(round(right_curvature, 0)):4d}m"

    out = np.copy(image)

    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    cv2.putText(out,
                text,
                (10, 15 + size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return out


def print_offset(image, offset):
    text = f"The car is {abs(offset):.3f}m to the {'right' if offset <= 0 else 'left'}"
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)

    out = np.copy(image)

    cv2.putText(out,
                text,
                (10, 50 + size[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return out


def get_offset(image, left_points, right_points, params):
    offset = (left_points[0][0] + right_points[0][0]) / 2 - image.shape[1] / 2
    offset = params['xm_per_pix'] * offset
    return offset


def image_pipeline(image, params, show_plots=False):
    undistorted = undistort(image, params)

    gradients = gradient_threshold(undistorted, params)
    colors = color_threshold(undistorted, params)

    combined_threshold = np.logical_or(gradients, colors)

    perspective, M, M_inv = find_perspective_lines(combined_threshold, params, show_plots=show_plots)
    out_img, left_points, right_points, left_fit, right_fit = fit_polynomial(perspective)

    reverse_transform = draw_lines(image, left_points, right_points, M_inv)

    y_eval = np.max(left_points[:, 1])
    left_curvature = calculate_curvature(left_fit, y_eval, params)
    right_curvature = calculate_curvature(right_fit, y_eval, params)

    offset = get_offset(reverse_transform, left_points, right_points, params)

    curvature_image = print_curvature(reverse_transform, left_curvature, right_curvature)
    offset_image = print_offset(curvature_image, offset)

    if show_plots:
        cv2.imshow('undistorted', undistorted)
        cv2.imshow('grads', convert_to_image(gradients))
        cv2.imshow('colors', convert_to_image(colors))
        cv2.imshow("Combined", convert_to_image(combined_threshold))
        cv2.imshow('Reverse Transform', reverse_transform)
        cv2.imshow('Sliding Lanes', out_img)
        cv2.imshow("Perspective", perspective)
        cv2.waitKey(500)

    return offset_image


def video_pipeline(video_location, output_folder, params, show_plots=False):
    file_name = os.path.basename(video_location)

    cap = cv2.VideoCapture(video_location)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(round(cap.get(cv2.CAP_PROP_FPS), 0))

    out = cv2.VideoWriter(f'{output_folder}/{file_name}', cv2.VideoWriter_fourcc(*'mp4v'), fps,
                          (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
