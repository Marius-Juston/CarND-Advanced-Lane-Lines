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


def image_pipeline(image, params):

    gradients = gradient_threshold(image, params)
    colors = color_threshold(image, params)

    cv2.imshow('undistorted', undistorted)
    cv2.imshow('grads', convert_to_image(gradients))
    cv2.imshow('colors', convert_to_image(colors))

    return np.logical_or(gradients, colors)
