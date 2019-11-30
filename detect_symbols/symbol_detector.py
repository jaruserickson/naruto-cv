"""
Symbol Detector
"""

import cyvlfeat.sift
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from . import symbols
from .plot_utils import plot_sift_keypoints


class SymbolDetector():
    def __init__(self):
        self._symbols = symbols.load_symbols()

    def process(self, frame):
        if frame is None:
            return None

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        n, m = frame_grey.shape

        if n > 1000 or m > 1000:
            print('Resolution too high!')
            return frame

        leaf = self._symbols[0]

        gx = cv2.Sobel(frame_grey, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame_grey, cv2.CV_64F, 0, 1, ksize=3)
        phi = np.arctan2(gy, gx)
        frame_grey = (frame_grey * 255).astype(np.uint8)
        edges = cv2.Canny(frame_grey, 20, 40)
        inds = np.argwhere(edges > 0)

        acc = np.zeros((n, m, 60, 36))

        for y, x in inds:
            # r = leaf.R(phi[i])[:, 0]
            # a = leaf.R(phi[i])[:, 1]
            # x = r * np.cos(a)
            # y = r * np.sin(a)

            for s in range(60):
                th = 0. 
                r = leaf.R(phi[y, x])[:, 0]
                a = leaf.R(phi[y, x])[:, 1]
                cx = x + r * s * np.cos(a)
                cy = y + r * s * np.sin(a)

                cx = np.clip(cx, a_min=0, a_max=m-1)
                cy = np.clip(cy, a_min=0, a_max=n-1)
                acc[cy.astype(np.int), cx.astype(np.int), s, 0] += 1
                # for j in range(36):
                #     cx = x - s * (x * math.cos(th) - y * math.sin(th))
                #     cy = y - s * (x * math.sin(th) + y * math.cos(th))
                #     acc[cy.astype(np.int), cx.astype(np.int), s, j] += 1
                #     th += 2 * np.pi / 36

        acc[:,0,:,:] = acc[:,m-1,:,:] = acc[0,:,:,:] = acc[n-1,:,:,:] = 0
        thresh = np.max(acc) - 1
        inds = np.argwhere(acc > thresh)
        print(f'count: {len(inds)}')

        if len(inds) < 1000:
            for i in range(len(inds)):
                cy, cx, s, th = inds[i]
                cv2.circle(frame, (cx, cy), s * 4, (0, 0, 255))

        return frame

        