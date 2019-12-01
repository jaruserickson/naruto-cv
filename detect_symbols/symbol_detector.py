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

        # to speed up process, remove need to check range of gradient
        for i in range(len(self._symbols)):
            if self._symbols[i]._R is not None:
                self._symbols[i]._R = self._symbols[i]._R + self._symbols[i]._R
                self._symbols[i]._R.append(self._symbols[i]._R[0])


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

        min_phi = symbols.MIN_GRAD * np.pi / 180
        d_phi = symbols.GRAD_STEP * np.pi / 180
        phi = ((phi - min_phi) / d_phi).astype(np.int)

        frame_grey = (frame_grey * 255).astype(np.uint8)
        edges = cv2.Canny(frame_grey, 20, 40)
        inds = np.argwhere(edges > 0)

        scale_steps = int((symbols.MAX_SCALE - symbols.MIN_SCALE) / symbols.SCALE_STEP) + 1
        theta_steps = int((symbols.MAX_ROTATION - symbols.MIN_ROTATION) / symbols.ROTATION_STEP) + 1
        acc = np.zeros((n, m, scale_steps, theta_steps), dtype=np.int32)

        for y, x in inds:
            cx = x + leaf.R(phi[y, x])[:, 1]
            cy = y + leaf.R(phi[y, x])[:, 0]
            s = leaf.R(phi[y, x])[:, 2]
            th = leaf.R(phi[y, x])[:, 3]

            cx = np.clip(cx, a_min=0, a_max=m-1).astype(np.int)
            cy = np.clip(cy, a_min=0, a_max=n-1).astype(np.int)
            acc[cy, cx, s, th] += 1

        acc[:,0,:,:] = acc[:,m-1,:,:] = acc[0,:,:,:] = acc[n-1,:,:,:] = 0

        ### uncomment to visualize accumulator
        # frame = np.max(acc, axis=(2,3))
        # frame = frame * 255.0 / np.max(frame) 
        # frame = np.dstack((frame,)*3).astype(np.uint8)

        thresh = 20
        cy, cx, s, th = np.unravel_index(np.argmax(acc), acc.shape)

        if acc[cy, cx, s, th] > thresh:
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255))
            w = symbols.MIN_SCALE + s * symbols.SCALE_STEP / 2
            th = (symbols.MIN_ROTATION + th * symbols.ROTATION_STEP) * np.pi / 180
            r = np.array([
                [math.cos(th), -math.sin(th)],
                [math.sin(th), math.cos(th)]
            ])
            p = np.array([
                [w, w],
                [-w, w],
                [-w, -w],
                [w, -w]
            ])
            p = np.matmul(r, p.T).T + [cx, cy]
            p = p.astype(np.int)
            cv2.line(frame, tuple(p[0]), tuple(p[1]), (0, 0, 255))
            cv2.line(frame, tuple(p[1]), tuple(p[2]), (0, 0, 255))
            cv2.line(frame, tuple(p[2]), tuple(p[3]), (0, 0, 255))
            cv2.line(frame, tuple(p[3]), tuple(p[0]), (0, 0, 255))

        return frame

        