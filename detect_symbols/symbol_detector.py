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

        min_th = -45
        max_th = 45
        dtheta = 10
        th_range = int((max_th - min_th) / 10)
        acc = np.zeros((n, m, 20, th_range), dtype=np.int32)

        for y, x in inds:
            # r = leaf.R(phi[y, x])[:, 0]
            # a = leaf.R(phi[y, x])[:, 1]
            # cx = x + r * np.cos(a)
            # cy = y + r * np.sin(a)
            # cx = np.clip(cx, a_min=0, a_max=m-1)
            # cy = np.clip(cy, a_min=0, a_max=n-1)
            # acc[cy.astype(np.int), cx.astype(np.int), 10, 0] += 1

            for s in range(1, 20):
                # r = leaf.R(phi[y, x])[:, 0]
                # a = leaf.R(phi[y, x])[:, 1]
                # cx = x + r * s / 10 * np.cos(a)
                # cy = y + r * s / 10 * np.sin(a)

                # cx = np.clip(cx, a_min=0, a_max=m-1)
                # cy = np.clip(cy, a_min=0, a_max=n-1)
                # acc[cy.astype(np.int), cx.astype(np.int), s, 0] += 1

                # tx = r * np.cos(a)
                # ty = r * np.sin(a)

                th = (min_th + dtheta / 2) * np.pi / 180
                for j in range(th_range):
                    new_phi = phi[y, x] - th
                    if new_phi < -np.pi:
                        new_phi += np.pi * 2
                    elif new_phi > np.pi:
                        new_phi -= np.pi * 2

                    r = leaf.R(new_phi)[:, 0]
                    a = leaf.R(new_phi)[:, 1]
                    cx = x + r * s / 10 * np.cos(th + a)
                    cy = y + r * s / 10 * np.sin(th + a)


                    # dx = x + r * np.cos(a)
                    # dy = y + r * np.sin(a)
                    # cx = x + s/10 * (dx * math.cos(th) - dy * math.sin(th))
                    # cy = y + s/10 * (dx * math.sin(th) + dy * math.cos(th))
                    cx = np.clip(cx, a_min=0, a_max=m-1).astype(np.int)
                    cy = np.clip(cy, a_min=0, a_max=n-1).astype(np.int)
                    acc[cy, cx, s, j] += 1
                    th += dtheta * np.pi / 180

        acc[:,0,:,:] = acc[:,m-1,:,:] = acc[0,:,:,:] = acc[n-1,:,:,:] = 0
        thresh = np.max(acc) - 1
        inds = np.argwhere(acc > thresh)
        print(f'count: {len(inds)}')

        # frame = np.max(acc, axis=(2,3))
        # frame = frame * 255.0 / np.max(frame) 
        # frame = np.dstack((frame,)*3).astype(np.uint8)

        if len(inds) < 1000:
            for i in range(len(inds)):
                cy, cx, s, th = inds[i]
                cv2.circle(frame, (cx, cy), 2, (0, 0, 255))
                w = s * 5
                th = (min_th + (th + .5) * dtheta) * np.pi / 180
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

        