""" Initialize symbols used for detection. """

import os
from skimage import io
import cyvlfeat.sift
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import symbols
from plot_utils import plot_sift_keypoints


def create_symbol(id, dphi, R):
    """ Create a symbol from a list of lists of phi mappings (R). """
    n = len(R)
    assert(n % 2 == 0)
    # make numpy array
    for i in range(n):
        R[i] = np.array(R[i])
    # concatenate phi with phi + pi (% 2pi)
    for i in range(n):
        R[i] = np.concatenate((R[i], R[int(i + n/2) % n]))
    # normalize radii to mean 4 pixels (this will be the minimum scale)
    for i in range(n):
        R[i][:, 0] = R[i][:, 0] * 4 / np.mean(R[i][:, 0])
    return symbols.Symbol(id, dphi, R)


class KPChooserGUI():
    """ Keypoint chooser GUI for initializing symbols. """
    def __init__(self, root_dir):
        self.ax_im = None
        self.ax = None
        self.id = -1
        self.n = 18
        self.R = [[] for _ in range(self.n)]
        self.dphi = np.pi * 2 / self.n
        self.symbols = [symbols.Symbol(i) for i in range(len(symbols.SYMBOL_IDS))]

        self.files = [None for i in range(len(self.symbols))]
        for fname in os.listdir(root_dir):
            basename = os.path.splitext(fname)[0]
            if basename.isdigit():
                id = int(basename)
                if id < len(self.files) and self.files[id] is None:
                    self.files[id] = os.path.join(root_dir, fname)

    def run(self):
        self.ax_im = plt.imshow(np.zeros((1,1)))
        self.ax = plt.gca()
        plt.axis('off')
        self.load_next_id()

        bsave = Button(plt.axes([0.82, 0.05, 0.1, 0.075]), 'Save')
        bsave.on_clicked(self.save)
        bnext = Button(plt.axes([0.71, 0.05, 0.1, 0.075]), 'Next')
        bnext.on_clicked(self.next)
        bprev = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Previous')
        bprev.on_clicked(self.prev)

        plt.show()
            
    def load_next_id(self, reverse=False):
        if reverse:
            offset = -1
            end = -1
        else:
            offset = 1
            end = len(self.symbols)

        while self.id + offset != end:
            self.id += offset

            if self.files[self.id] is None:
                print(f'Couldnt find image for symbol index {self.id}. Skipping.')
            else:
                img = io.imread(self.files[self.id], as_gray=True).astype(np.float32)

                n, m = img.shape
                cy, cx = int(n / 2), int(m / 2)
                gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
                gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
                phi = np.arctan2(gy, gx)
                img = (img * 255).astype(np.uint8)
                edges = cv2.Canny(img, 10, 20)
                display = np.zeros((n, m, 3))
                dist_thresh = 3 * cx / 4

                self.R = [[] for _ in range(self.n)]

                for i in range(n):
                    for j in range(m):
                        if edges[i, j] > 0:
                            r = np.sqrt((cx-j)**2 + (cy-i)**2)

                            if r < dist_thresh:
                                a = np.arctan2((cy-i), (cx-j))
                                self.R[int(np.pi + phi[i, j] / self.dphi)].append([r, a])

                                if phi[i, j] < 0:
                                    phi[i, j] += np.pi
                                display[i, j] = (.5 + np.cos(phi[i, j]) * .5, 
                                                0, 
                                                .5 + np.sin(phi[i, j]) * .5)
                
                cv2.circle(display, (cx, cy), 2, (0., 1., 0.))
                self.ax_im.set_data(display)
                self.ax.set_title('Symbol ' + str(self.id))
                plt.draw()

                return True

        # plt.close()
        print('Reached end of symbols')
        return False

    def next(self, _):
        self.load_next_id()

    def prev(self, _):
        self.load_next_id(reverse=True)

    def save(self, _):
        self.symbols[self.id] = create_symbol(self.id, self.dphi, self.R)
        print(f'Saved symbol {self.id}.')

    def write(self):
        symbols.save_symbols(self.symbols)
        print('All symbols saved to file.')


def initialize_symbols(folder='symbols'):
    """ Go through the symbol images in the given folder, extract their feature 
    points, and save them to file. 
    """
    callback = KPChooserGUI('./symbols')
    
    callback.run()
    callback.write()


if __name__ == '__main__':
    initialize_symbols()
