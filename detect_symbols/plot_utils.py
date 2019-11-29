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

        cv2.circle(img, (x, y), int(round(2 * s)), color, 2)