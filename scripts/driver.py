import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip

from scripts.pipeline import image_pipeline


def process_images(image_folder, output_folder, params):
    for file in os.listdir(image_folder):
        image = cv2.imread(f'{image_folder}/{file}')
        out = image_pipeline(image, params)
        cv2.imwrite(f'{output_folder}/{file}', out)
        cv2.waitKey(500)

    cv2.destroyAllWindows()


def process_video(video, output_folder, params):
    clip1 = VideoFileClip(video)
    white_clip = clip1.fl_image(
        lambda image: cv2.cvtColor(image_pipeline(cv2.cvtColor(image, cv2.COLOR_RGB2BGR), params), cv2.COLOR_BGR2RGB))
    white_clip.write_videofile(f'{output_folder}/1.mp4', audio=False)


if __name__ == '__main__':
    params = {
        'k_size_x': 7,
        'k_size_y': 7,
        'k_size_mag': 9,
        'k_size_dir': 17,
        'min_x': 31,
        'min_y': 37,
        'min_mag': 42,
        'min_dir': np.pi / 2 * 42 / 100,
        'max_x': 102,
        'max_y': 116,
        'max_mag': 109,
        'max_dir': np.pi / 2 * 81 / 100,
        'hsl_min': (0, 98, 90),
        'hsl_max': (81, 255, 255),
        'image_calib_folder': '../camera_cal',
        'chessboard_size': (9, 6),
        'undistort': 'undistort.pickle',
        'xm_per_pix': 3.7 / 700,
        'ym_per_pix': 30 / 720
    }

    process_images('../test_images', '../output_images', params)
