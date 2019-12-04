""" Filtering functions.
"""

import numpy as np


def filter_col(img, f, mode='same', **kwargs):
    """ Filter an image with a single column filter.

    Args:
        - img: input image (2D numpy array)
        - f: input filter (1D numpy array)
        - mode: indicates size of output image ('full', 'same', or 'valid')
        - kwargs: keyword arguments for padding. If mode is not 'valid', pad 
        mode can be specified using 'pad-type'
    """
    k = len(f)

    if mode == 'full':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((k-1, k-1), (0, 0)), kwargs['pad_type'], **kwargs)
        else:
            img_pad = np.pad(img, ((k-1, k-1), (0, 0)), **kwargs)
    elif mode == 'same':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((int(k/2), int(k/2)), (0, 0)), kwargs['pad_type'], **kwargs)
        else:
            img_pad = np.pad(img, ((int(k/2), int(k/2)), (0, 0)), **kwargs)
    elif mode == 'valid':
        img_pad = img
    else:
        raise ValueError('Mode not a valid filter mode.')

    n, m = img_pad.shape
    layers = [img_pad[i:n-k+i+1, :].flatten() for i in range(k)]
    img_mat = np.stack(layers, axis=0)
    return np.matmul(f, img_mat).reshape((n-k+1, m))


def filter_row(img, f, mode='same', **kwargs):
    """ Filter an image with a single row filter.

    Args:
        - img: input image (2D numpy array)
        - f: input filter (1D numpy array)
        - mode: indicates size of output image ('full', 'same', or 'valid')
        - kwargs: keyword arguments for padding. If mode is not 'valid', pad 
        mode can be specified using 'pad-type'
    """
    k = len(f)

    if mode == 'full':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((0, 0), (k-1, k-1)), mode=kwargs.pop('pad_type'), **kwargs)
        else:
            img_pad = np.pad(img, ((0, 0), (k-1, k-1)), **kwargs)
    elif mode == 'same':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((0, 0), (int(k/2), int(k/2))), mode=kwargs.pop('pad_type'), **kwargs)
        else:
            img_pad = np.pad(img, ((0, 0), (int(k/2), int(k/2))), **kwargs)
    elif mode == 'valid':
        img_pad = img
    else:
        raise ValueError('Mode not a valid filter mode.')

    n, m = img_pad.shape
    layers = [img_pad[:, i:m-k+i+1].flatten() for i in range(k)]
    img_mat = np.stack(layers, axis=0)
    return np.matmul(f, img_mat).reshape((n, m-k+1))


def filter_sep(img, fx, fy, mode='same', **kwargs):
    """ Filter an image with the given filter components.

    Args:
        - img: input image (2D numpy array)
        - fx: input row filter (1D numpy array)
        - fy: input column filter (1D numpy array)
        - mode: indicates size of output image ('full', 'same', or 'valid')
        - kwargs: keyword arguments for padding. If mode is not 'valid', pad 
        mode can be specified using 'pad-type'
    """
    kx = len(fx)
    ky = len(fy)

    if mode == 'full':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((ky-1, ky-1), (kx-1, kx-1)), mode=kwargs.pop('pad_type'), **kwargs)
        else:
            img_pad = np.pad(img, ((ky-1, ky-1), (kx-1, kx-1)), **kwargs)
    elif mode == 'same':
        if 'pad_type' in kwargs:
            img_pad = np.pad(img, ((int(ky/2), int(ky/2)), (int(kx/2), int(kx/2))), mode=kwargs.pop('pad_type'), **kwargs)
        else:
            img_pad = np.pad(img, ((int(ky/2), int(ky/2)), (int(kx/2), int(kx/2))), **kwargs)
    elif mode == 'valid':
        img_pad = img
    else:
        raise ValueError('Mode not a valid filter mode.')

    temp = filter_col(img_pad, fy, mode='valid', **kwargs)
    return filter_row(temp, fx, mode='valid', **kwargs)


if __name__ == '__main__':
    # filtering example
    img = np.arange(9).reshape((3,3))
    fx = np.array([0,0,1])
    fy = np.array([0,0,1])
    out = filter_sep(img, fx, fy, mode='same', pad_type='constant', constant_values=0)
    print(out)
