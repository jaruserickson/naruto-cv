"""
Symbol Detector
"""

import cyvlfeat.sift
import cv2
import numpy as np
import matplotlib.pyplot as plt
from . import symbols
from .plot_utils import plot_sift_keypoints, draw_bounding_box
import math


def match_features_to_predictions(img, kp, desc, symbol, thresh=5000):
    for i in range(len(kp)):
        dists = np.sum((desc[i] - symbol.desc)**2, axis=1)

        if np.min(dists) < thresh:
            j = np.argmin(dists)
            th = kp[i, 3] - symbol.kp[j, 2]
            s = kp[i, 2] / symbol.kp[j, 1]
            x = kp[i, 1] + symbol.kp[j, 0] * s * np.cos(th - symbol.kp[j, 3])
            y = kp[i, 0] + symbol.kp[j, 0] * s * np.sin(th - symbol.kp[j, 3])

            x, y = int(x), int(y)

            plot_sift_keypoints(img, np.array([y, x, s * 10, th]), (255, 0, 0))
            plot_sift_keypoints(img, kp[i], (0, 0, 255))
            cv2.line(img, (x, y), (kp[i,1], kp[i,0]), (255, 0, 255))


class SymbolDetector():
    def __init__(self):
        self._symbols = symbols.load_symbols()

    def process(self, frame):
        if frame is None:
            return None

        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect keypoints and match descriptors
        kp, desc = cyvlfeat.sift.sift(
            frame_grey, 
            compute_descriptor=True,
            peak_thresh=10)

        n, m = frame_grey.shape
        min_scale = 10
        max_scale = 200
        d_scale = 10
        min_theta = 0 * np.pi / 180
        max_theta = 360 * np.pi / 180 
        d_theta = 5 * np.pi / 180 
        scale_steps = int((max_scale - min_scale) / d_scale) + 1
        theta_step = int((max_theta - min_theta) / d_theta) + 1
        acc = np.zeros((n, m, scale_steps, theta_step), dtype=np.int16)

        leaf = self._symbols[0]

        ### uncomment to see each detected feature's best prediction
        # match_features_to_predictions(frame, kp, desc, leaf, thresh=10000)
        # return frame

        dists = np.empty((len(kp), len(leaf.kp)))
        inds = np.empty((len(kp), len(leaf.kp), 4))

        for i in range(len(leaf.kp)):
            dists[:, i] = np.sum((desc - leaf.desc[i])**2, axis=1)
            th = kp[:, 3] - leaf.kp[i, 2]
            s = kp[:, 2] / leaf.kp[i, 1]
            cx = kp[:, 1] + leaf.kp[i, 0] * s * np.cos(th - leaf.kp[i, 3])
            cy = kp[:, 0] + leaf.kp[i, 0] * s * np.sin(th - leaf.kp[i, 3])

            th[th < 0] += np.pi * 2
            th[th < 0] -= np.pi * 2

            inds[:, i, 0] = cy
            inds[:, i, 1] = cx
            inds[:, i, 2] = (s * 100 - min_scale) / d_scale
            inds[:, i, 3] = (th - min_theta) / d_theta
        
        thresh = 5000
        inds = inds[dists < thresh].astype(np.int)
        inds = np.clip(inds, a_min=0, a_max=[n-1, m-1, 9, 35])
        
        for i in range(len(inds)):
            acc[inds[i,0], inds[i,1], inds[i,2], inds[i,3]] += 1

        thresh = 0
        inds = np.argwhere(acc > thresh)
        max_a = len(leaf.kp)

        for cy, cx, k, h in inds:
            s = min_scale + k * d_scale
            th = min_theta + h * d_theta
            conf = acc[cy, cx, k, h] / max_a

            w = s
            th = -th
            r = np.array([
                [math.cos(th), -math.sin(th)],
                [math.sin(th), math.cos(th)]
            ])
            p = np.array([
                [-w, -w],
                [-w, w],
                [w, w],
                [w, -w]
            ])
            p = np.matmul(r, p.T).T + [cx, cy]
            p = p.astype(np.int)

            color = (255, 255, 100 + int(conf * 155))
            draw_bounding_box(frame, tuple(p[0]), tuple(p[1]), tuple(p[2]), tuple(p[3]), (255, 0, 0))
            # plot_sift_keypoints(frame, np.array([y, x, s * 10, th]), color)

        #plot_sift_keypoints(frame, kp, (0, 0, 255))

        return frame

        