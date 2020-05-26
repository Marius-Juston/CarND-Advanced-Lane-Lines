import cv2
import numpy as np

from scripts.pipeline import image_pipeline, convert_to_image

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
        'chessboard_size': (9, 5),
        'undistort': 'undistort.pickle'
    }

    image = cv2.imread('../test_images/test2.jpg')

    out = image_pipeline(image, params)

    cv2.imshow("Ouput", convert_to_image(out))

    cv2.waitKey()
    cv2.destroyAllWindows()
