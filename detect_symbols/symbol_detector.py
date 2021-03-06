"""
Symbol Detector.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from . import symbols
import time
import imgproc


class SymbolDetector():
    """ The symbol detector class. """
    
    def __init__(self):
        self._symbols = symbols.load_symbols()
        
        # to speed up process, remove need to check range of gradient
        for i in range(len(self._symbols)):
            if self._symbols[i]._R is not None:
                self._symbols[i]._R = self._symbols[i]._R + self._symbols[i]._R
                self._symbols[i]._R.append(self._symbols[i]._R[0])


    def process(self, frame, sym_id):
        """ Process one frame, or one image 
        
        Args:
            - frame: input image
            - sym_id: symbol id to detect
        """
        p, score = None, 0

        if frame is None:
            print('Invalid frame')
            return p, score

        if sym_id < 0 or sym_id >= len(symbols.SYMBOL_IDS):
            print('Invalid symbol')
            return p, score

        time_start = time.clock()
        n, m, _ = frame.shape
        fsize = 1.

        # check size
        max_size = 200

        if n > max_size or m > max_size:
            if n > m:
                fsize = max_size / n
                m = int(m * fsize)
                n = max_size
            else:
                fsize = max_size / m
                n = int(n * fsize)
                m = max_size
            frame = cv2.resize(frame, (m, n), interpolation=cv2.INTER_LINEAR)

        # Canny 
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_grey = (frame_grey * 255).astype(np.uint8)
        frame_grey = imgproc.gaussian_blur(frame_grey, 5, 1.2)
        edges, phi = imgproc.canny(frame_grey, 20, 40)

        min_phi = symbols.MIN_GRAD * np.pi / 180
        d_phi = symbols.GRAD_STEP * np.pi / 180
        phi = ((phi - min_phi) / d_phi).astype(np.int)

        # Hough
        scale_steps = int((symbols.MAX_SCALE - symbols.MIN_SCALE) / symbols.SCALE_STEP) + 1
        theta_steps = int((symbols.MAX_ROTATION - symbols.MIN_ROTATION) / symbols.ROTATION_STEP) + 1
        acc = np.zeros((n, m, scale_steps, theta_steps), dtype=np.int32)

        sym = self._symbols[sym_id]
        inds = np.argwhere(edges > 0)

        for y, x in inds:
            cx = x + sym.R(phi[y, x])[:, 1]
            cy = y + sym.R(phi[y, x])[:, 0]
            s = sym.R(phi[y, x])[:, 2]
            th = sym.R(phi[y, x])[:, 3]

            cx = np.clip(cx, a_min=0, a_max=m-1).astype(np.int)
            cy = np.clip(cy, a_min=0, a_max=n-1).astype(np.int)
            acc[cy, cx, s, th] += 1

        acc[:,0,:,:] = acc[:,m-1,:,:] = acc[0,:,:,:] = acc[n-1,:,:,:] = 0
        cy, cx, s, th = np.unravel_index(np.argmax(acc), acc.shape)
        score = acc[cy, cx, s, th] / sym.num_edges
        thresh = 0.0

        # get bounding box
        if score > thresh:
            w = symbols.MIN_SCALE + s * symbols.SCALE_STEP / 2
            th = (symbols.MIN_ROTATION + th * symbols.ROTATION_STEP) * np.pi / 180
            cx /= fsize
            cy /= fsize
            w /= fsize

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

            print(f'Found {symbols.SYMBOL_IDS[sym_id]} with score {score}')
        else:
            print('No symbol found')

        # ### uncomment to visualize accumulator
        # display = np.max(acc, axis=(2,3))
        # display = display * 255.0 / np.max(display) 
        # display = np.dstack((display,)*3).astype(np.uint8)
        # cv2.imshow('Accumulator', display)
        # cv2.waitKey(0)

        time_passed = time.clock() - time_start
        print(f'Time elapsed: {time_passed}')
        return p, score
