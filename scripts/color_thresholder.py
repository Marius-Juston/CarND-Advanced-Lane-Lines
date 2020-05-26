import cv2


def hsl_select(img, min_thresh=(0, 0, 0), max_threshold=(180, 255, 255)):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hsl, min_thresh, max_threshold)

    return mask


def rgb_select(img, min_thresh=(0, 0, 0), max_threshold=(180, 255, 255)):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask = cv2.inRange(hsl, min_thresh, max_threshold)

    return mask


def hsv_select(img, min_thresh=(0, 0, 0), max_threshold=(180, 255, 255)):
    hsl = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsl, min_thresh, max_threshold)

    return mask
