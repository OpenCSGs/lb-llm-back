"""Object-detection conversion helpers."""

import cv2
import numpy as np


def mask_to_rectangle(mask):
    points = cv2.findNonZero((mask > 0).astype(np.uint8))
    if points is None:
        return None
    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = mask.shape[:2]
    return {
        'x': round(x / image_width * 100, 4),
        'y': round(y / image_height * 100, 4),
        'width': round(width / image_width * 100, 4),
        'height': round(height / image_height * 100, 4),
    }
