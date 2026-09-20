"""Polygon semantic-segmentation conversion helpers."""

import os

import cv2
import numpy as np


def mask_to_polygons(mask):
    min_area = float(os.getenv('VOLCENGINE_POLYGON_MIN_AREA', '50'))
    simplify = float(os.getenv('VOLCENGINE_POLYGON_SIMPLIFY_PX', '2.0'))
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    height, width = mask.shape[:2]
    polygons = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        points = cv2.approxPolyDP(contour, simplify, True)
        if len(points) < 3:
            continue
        polygons.append([
            [round(float(point[0][0]) / width * 100, 4),
             round(float(point[0][1]) / height * 100, 4)]
            for point in points
        ])
    return polygons
