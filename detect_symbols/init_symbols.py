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


def create_symbol(id, kp, desc):
    """ Create a symbol from a set of keypoints and descriptors. """
    y, x, s, th = kp[:, 0], kp[:, 1], kp[:, 2], kp[:, 3]
    cx = np.mean(x)
    cy = np.mean(y)
    r = np.sqrt((cx - x)**2 + (cy - y)**2)
    phi = th
    alpha = np.arctan2(y - cy, cx - x)
    kp = np.stack((r, s, phi, alpha), axis=1)
    return symbols.Symbol(id, kp, desc)


class KPChooserGUI():
    """ Keypoint chooser GUI for initializing symbols. """
    def __init__(self, root_dir):
        self.ax_im = None
        self.ax = None
        self.img = None
        self.id = -1
        self.ind = 0
        self.kp = None
        self.desc = None
        self.chosen = None
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

        bdone = Button(plt.axes([0.82, 0.05, 0.1, 0.075]), 'Done')
        bdone.on_clicked(self.done)
        bnext = Button(plt.axes([0.71, 0.05, 0.1, 0.075]), 'Next')
        bnext.on_clicked(self.next)
        bval = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Choose')
        bval.on_clicked(self.choose)
        bprev = Button(plt.axes([0.49, 0.05, 0.1, 0.075]), 'Previous')
        bprev.on_clicked(self.prev)

        plt.show()
            
    def load_next_id(self):
        self.id += 1

        while self.id < len(self.symbols):
            if self.files[self.id] is not None:
                self.img = io.imread(self.files[self.id], as_gray=True)
                self.kp, self.desc = cyvlfeat.sift.sift(self.img, compute_descriptor=True)
                self.chosen = np.zeros(len(self.kp), dtype=np.int)
                self.ind = 0
                
                self.img = cv2.cvtColor(self.img.astype(np.float32), cv2.COLOR_GRAY2RGB)
                self.img = (self.img * 255.).astype(np.uint8)
                display = self.img.copy()
                plot_sift_keypoints(display, self.kp[self.ind], color=(255, 0, 0))
                self.ax_im.set_data(display)
                self.ax.set_title('Symbol ' + str(self.id))
                plt.draw()

                return True
            
            print(f'Couldnt find image for symbol index {self.id}. Skipping.')
            self.id += 1

        plt.close()
        return False

    def next(self, _):
        if self.ind + 1 < len(self.kp):
            self.ind += 1
            print(f'{self.ind}/{len(self.kp)}')
            
            display = self.img.copy()
            plot_sift_keypoints(display, self.kp[self.chosen == 1], color=(100,100,255))
            plot_sift_keypoints(display, self.kp[self.ind], color=(255, 0, 0))
            self.ax_im.set_data(display)
            plt.draw()
        else:
            print('Reached end of feature points.')

    def choose(self, _):
        if (self.chosen[self.ind] == 0):    
            self.chosen[self.ind] = 1
            print(f'Chose keypoint {self.ind}')
        else:
            self.chosen[self.ind] = 0
            print(f'Removed keypoint {self.ind}')

    def prev(self, _):
        if self.ind - 1 > 0:
            self.ind -= 1
            print(f'{self.ind}/{len(self.kp)}')
            
            display = self.img.copy()
            plot_sift_keypoints(display, self.kp[self.chosen == 1], color=(100,100,255))
            plot_sift_keypoints(display, self.kp[self.ind], color=(255, 0, 0))
            self.ax_im.set_data(display)
            plt.draw()
        else:
            print('Reached beginning of feature points.')

    def done(self, _):
        self.symbols[self.id] = create_symbol(self.id, self.kp[self.chosen == 1], self.desc[self.chosen == 1])
        print(f'Saved {len(self.symbols[self.id].kp)} keypoints for symbol {self.id}.')
        self.load_next_id()

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
