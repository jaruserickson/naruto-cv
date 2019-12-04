""" Canny implementation.
"""

import numpy as np
from . import filter
from . import gaussian


CANNY_DEFAULT_APERTURE = 3


def canny(img, threshold1, threshold2, aperture=None, sigma=1.4):
    """ Detect canny edges, and return the edge gradient directions along with the edges.

    Args:
        - img: input grey image (2D numpy array)
        - threshold1: low threshold used for hysteresis
        - threshold2: high threshold used for hysteresis
        - aperture: size of Sobel operator
        - sigma: Gaussian parameter
    """

    if aperture == None:
        aperture = CANNY_DEFAULT_APERTURE

    if aperture != CANNY_DEFAULT_APERTURE:
        raise UserWarning(f'Only aperture of {CANNY_DEFAULT_APERTURE} is implemented')

    dtype = np.float64

    # Sobel
    f1 = np.array([-1, 0, 1], dtype=dtype)
    f2 = np.array([1, 2, 1], dtype=dtype)
    gx = filter.filter_sep(img, f1, f2, mode='same', pad_type='reflect')
    gy = filter.filter_sep(img, f2, f1, mode='same', pad_type='reflect')

    # compute gradient
    grad_dir = np.arctan2(gy, gx)
    grad_mag = np.abs(gx) + np.abs(gy)
    grad_dir = np.pad(grad_dir, 1, mode='constant', constant_values=0)
    grad_mag = np.pad(grad_mag, 1, mode='constant', constant_values=0)
    threshold1 *= 2
    threshold2 *= 2

    # non maxima suppression
    grad_dx = np.round(np.cos(grad_dir)).astype(np.int)
    grad_dy = np.round(np.sin(grad_dir)).astype(np.int)
    grad_dx[:,0] = grad_dx[:,-1] = grad_dx[0,:] = grad_dx[-1,:] = 0
    grad_dy[:,0] = grad_dy[:,-1] = grad_dy[0,:] = grad_dy[-1,:] = 0
    coords = np.indices(grad_mag.shape)

    grad_mag[grad_mag < grad_mag[coords[0] + grad_dy, coords[1] + grad_dx]] = 0
    grad_mag[grad_mag < grad_mag[coords[0] - grad_dy, coords[1] - grad_dx]] = 0

    # hysteresis
    q = np.zeros((grad_mag.shape[0] * grad_mag.shape[1], 2), dtype=np.int)
    strong = np.argwhere(grad_mag > threshold2)
    grad_mag[strong[:, 0], strong[:, 1]] = 0
    i = 0
    end = len(strong)
    q[:end] = strong

    while i < end:
        y, x = q[i][0], q[i][1]
        p1 = (y + grad_dx[y, x], x - grad_dy[y, x])
        p2 = (y - grad_dx[y, x], x + grad_dy[y, x])

        if grad_mag[p1] > threshold1:
            q[end] = p1
            grad_mag[p1] = 0
            end += 1
            
        if grad_mag[p2] > threshold1:
            q[end] = p2
            grad_mag[p2] = 0
            end += 1

        i += 1

    edges = np.zeros(img.shape, dtype=np.uint8)
    inds = q[:end] - [1, 1]
    edges[inds[:, 0], inds[:, 1]] = 255

    return edges, grad_dir[1:-1, 1:-1]

if __name__ == '__main__':
    # filtering example
    img = np.array([0,0,255,0,0], np.uint8)
    img = np.stack((img,)*5, axis=0)
    img = gaussian.gaussian_blur(img, 3, 1.)
    edges, phi = canny(img, 10, 20)
    #import cv2
    #edges = cv2.Canny(img, 10,20)
    print(edges)