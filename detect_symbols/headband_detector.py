"""
Headband Detector
"""

import cyvlfeat.sift
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from . import symbols
import time
import skimage


def connect_lines(lines, edges):
    return None


class HeadbandDetector():
    def __init__(self):
        a=0

    def process(self, frame):
        bnd_boxes = []

        if frame is None:
            print('Invalid frame')
            return bnd_boxes
            
        time_start = time.clock()
        n, m, _ = frame.shape

        # Canny 
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        gx = cv2.Sobel(frame_grey, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame_grey, cv2.CV_64F, 0, 1, ksize=3)
        phi = np.arctan2(gy, gx)

        frame_grey = (frame_grey * 255).astype(np.uint8)
        edges = cv2.Canny(frame_grey, 20, 40)

        # Hough
        max_theta = symbols.MAX_ROTATION * np.pi / 180
        min_theta = symbols.MIN_ROTATION * np.pi / 180
        d_theta = 1 * np.pi / 180
        theta_steps = int((max_theta - min_theta) / d_theta) + 1
        max_y = n + m
        min_y = -m
        y_steps = int(max_y - min_y) + 1
        tol = 15
        #theta = np.array([th * np.pi / 180 for th in range(theta_min, theta_max, d_theta)])
        #lines = skimage.transform.probabilistic_hough_line(edges, threshold=10, line_length=20, line_gap=10, theta=theta)
        weight = np.matmul(cv2.getGaussianKernel(tol, 0).T, cv2.getGaussianKernel(tol, 0))

        acc = np.zeros((y_steps + tol - 1, theta_steps + tol - 1), dtype=np.float32)

        for i in range(n):
            for j in range(m):
                if edges[i, j] > 0:
                    th = phi[i, j] - np.pi / 2

                    if th < -np.pi / 2:
                        th += np.pi
                    elif th > np.pi / 2:
                        th -= np.pi

                    if th < min_theta or th > max_theta:
                        continue
                        
                    k = int((th - min_theta) / d_theta)
                    y_int = i - j * math.tan(th)
                    h = int(y_int - min_y)
                    acc[h: h + tol, k: k + tol] += weight

        off = int(tol / 2)
        acc = acc[off: -off, off: -off]
        # display = (255 * acc / acc.max()).astype(np.uint8)
        # cv2.imshow('Acc', display)
        # cv2.waitKey(0)

        # find best lines
        thresh = 3 * np.max(acc) / 4 # m / 20
        lines = []
        cur_max = np.max(acc)

        while cur_max > thresh:
            h, k = np.unravel_index(np.argmax(acc), acc.shape)
            y = min_y + h
            th = min_theta + k * d_theta
            cur_max = acc[h, k]
            p1 = (0, int(y))
            p2 = (m, int(y + m * math.tan(th)))
            lines.append((p1, p2))

            h1 = max(0, h-tol+1)
            h2 = min(y_steps, h+tol)
            k1 = max(0, k-tol+1)
            k2 = min(theta_steps, k+tol)
            acc[h1:h2, k1:k2] = 0

        # find pairs of lines with grey between them
        #grey = frame[:, :, 0] - frame[:, :, 0]

        time_passed = time.clock() - time_start
        print(f'Time elapsed: {time_passed}')
        return lines
