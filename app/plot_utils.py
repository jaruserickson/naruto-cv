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
    cv2.line(img, p1, p2, color)
    cv2.line(img, p2, p3, color)
    cv2.line(img, p3, p4, color)
    cv2.line(img, p4, p1, color)

    if title is not None:
        if color[0] + color[1] + color[2] > 255 * 3 / 2:
            text_col = (0, 0, 0)
        else:
            text_col = (255, 255, 255)
        text_size = cv2.getTextSize(title, 0, 1, 1)

        if p1[1] - text_size[0][1] > 0:
            text_orig = p1
        else:
            text_orig = p4
        cv2.rectangle(img, text_orig, (text_orig[0] + text_size[0][0], text_orig[1] - text_size[0][1]), color, -1)
        cv2.putText(img, title, text_orig, 0, 1, text_col)
