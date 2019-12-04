""" Gaussian filter function
"""

import numpy as np
import collections
from . import filter


def get_gaussian_sep(ksize, sigma):
    """
    Create a 1D Gaussian kernel.

    Args:
        - param ksize: length of kernel (int)
        - param sigma: Gaussian parameter
    """
    result = np.empty(ksize, dtype=np.float)
    r = int((ksize + 1) / 2)

    for i in range(r):
        result[r-i-1] = result[r+i-1] = np.exp(-.5 * (i/sigma)**2)
    
    total = np.sum(result)
    return result / total


def gaussian_blur(img, ksize, sigma, dtype=None):
    """
    Gaussian blur.

    Args:
        - param ksize: length of kernel (int)
        - param sigma: Gaussian parameter
    """
    if dtype == None:
        dtype = img.dtype

    g = get_gaussian_sep(ksize, sigma)
    return filter.filter_sep(img, g, g).astype(dtype)


if __name__ == '__main__':
    f = get_gaussian_sep(5, 1.4)
    print(f)
    print(np.sum(f))