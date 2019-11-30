""" Plotting function for symbol detection. """

import cv2
import numpy as np


def plot_sift_keypoints(img, kp, color):
    h, w, _ = img.shape

    if len(kp.shape) < 2:
        kp = kp[np.newaxis, ...]
    
    for i in range(kp.shape[0]):
        y, x, s, th = kp[i]
        x = int(x)
        y = int(y)
        r = int(round(2 * s))

        cv2.circle(img, (x, y), r, color, 2)
        cv2.line(img, (x, y), (int(x + r * np.cos(th)), int(y + r * np.sin(th))), color)