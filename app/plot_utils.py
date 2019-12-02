""" Plotting function for symbol detection. """

import cv2
import numpy as np


def plot_sift_keypoints(img, kp, color, thickness=1):
    h, w, _ = img.shape

    if len(kp.shape) < 2:
        kp = kp[np.newaxis, ...]
    
    for i in range(kp.shape[0]):
        y, x, s, th = kp[i]
        x = int(x)
        y = int(y)
        r = int(round(2 * s))

        cv2.circle(img, (x, y), r, color, thickness)
        cv2.line(img, (x, y), (int(x + r * np.cos(th)), int(y + r * np.sin(th))), color, thickness)


def draw_bounding_box(img, p1, p2, p3, p4, color, title=None):
    thickness = 2
    cv2.line(img, p1, p2, color, thickness)
    cv2.line(img, p2, p3, color, thickness)
    cv2.line(img, p3, p4, color, thickness)
    cv2.line(img, p4, p1, color, thickness)

    if title is not None:
        cv2.putText(img, title, p1, 0, 1, color, thickness)
