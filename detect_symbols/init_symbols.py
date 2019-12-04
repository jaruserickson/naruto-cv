""" Initialize symbols used for detection. """

import os
from skimage import io
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import symbols

    
def create_symbol(id, R, num_edges, w=100):
    """ Create a symbol from a list of lists of phi mappings (R). """
    phi_steps = int((symbols.MAX_GRAD - symbols.MIN_GRAD) / symbols.GRAD_STEP / 2)
    scale_steps = int((symbols.MAX_SCALE - symbols.MIN_SCALE) / symbols.SCALE_STEP) + 1
    theta_steps = int((symbols.MAX_ROTATION - symbols.MIN_ROTATION) / symbols.ROTATION_STEP) + 1

    phi_min = symbols.MIN_GRAD * np.pi / 180
    d_phi = symbols.GRAD_STEP * np.pi / 180
    scale_min = symbols.MIN_SCALE
    d_scale = symbols.SCALE_STEP
    theta_min = symbols.MIN_ROTATION * np.pi / 180
    d_theta = symbols.ROTATION_STEP * np.pi / 180

    # place all mappings in first half of table 
    for i in range(phi_steps):
        R[i] = R[i] + R[int(i + phi_steps)]
    R[0] = R[0] + R[phi_steps - 1]

    # make numpy array
    for i in range(phi_steps):
        if len(R[i]) == 0:
            R[i] = np.empty((0,2))
        else:
            R[i] = np.array(R[i])

    new_R = [[] for _ in range(phi_steps)]

    # calculate indeces
    phi = phi_min + d_phi / 2
    for i in range(phi_steps):
        s = scale_min + d_scale / 2
        for j in range(scale_steps):

            th = theta_min + d_theta / 2
            for k in range(theta_steps):
                phi_prime = phi - th
                i_prime = int((phi_prime - phi_min) / d_phi)
                i_prime %= phi_steps

                r = R[i_prime][:, 0]
                a = R[i_prime][:, 1]
                cx = r * s / w * np.cos(th + a)
                cy = r * s / w * np.sin(th + a)
                cx = cx.astype(np.int)
                cy = cy.astype(np.int)

                for ii in range(len(cy)):
                    new_R[i].append([cy[ii], cx[ii], j, k])

                th += d_theta
            s += d_scale
        
        new_R[i] = np.array(new_R[i])
        phi += d_phi
    
    return symbols.Symbol(id, new_R, num_edges)


class InitSymbolGUI():
    """ GUI for initializing symbols. """
    def __init__(self, root_dir):
        self.ax_im = None
        self.ax = None
        self.id = -1
        self.symbols = symbols.load_symbols()
        print('All symbols loaded from file.')

        self.phi_steps = int((symbols.MAX_GRAD - symbols.MIN_GRAD) / symbols.GRAD_STEP) + 1
        self.min_phi = symbols.MIN_GRAD * np.pi / 180
        self.d_phi = symbols.GRAD_STEP * np.pi / 180
        self.R = [[] for _ in range(self.phi_steps)]
        self.num_edges = 0

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
                edges = cv2.Canny(img, 100, 120)
                display = np.zeros((n, m, 3))
                dist_thresh = 3 * cx / 4
                inds = np.argwhere(edges > 0)

                self.R = [[] for _ in range(self.phi_steps)]
                self.num_edges = len(inds)

                for i, j in inds:
                    r = np.sqrt((cx-j)**2 + (cy-i)**2)

                    if r < dist_thresh:
                        a = np.arctan2((cy-i), (cx-j))
                        self.R[int((phi[i, j] - self.min_phi) / self.d_phi)].append([r, a])

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
        self.symbols[self.id] = create_symbol(self.id, self.R, self.num_edges)
        print(f'Saved symbol {self.id}.')

    def write(self):
        symbols.save_symbols(self.symbols)
        print('All symbols saved to file.')


def initialize_symbols(folder='symbols'):
    """ Go through the symbol images in the given folder, extract their feature 
    points, and save them to file. 
    """
    callback = InitSymbolGUI('./symbols')
    
    callback.run()
    callback.write()


if __name__ == '__main__':
    initialize_symbols()
