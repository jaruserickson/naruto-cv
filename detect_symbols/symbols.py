"""
symbols.py

Anything that has to do with the symbols is in this file.
"""

import json
import os
from collections import defaultdict
import numpy as np
import cv2
import _pickle as pickle


SYMBOL_IDS = {
    0 : "leaf",
    1 : "sand",
    2 : "cloud",
    3 : "stone",
    4 : "mist"
}


def load_symbols(filename='symbols.pkl'):
    if os.path.isfile(filename):
        with open(filename, 'rb') as infile:
            symbols = pickle.load(infile)
    else:
        symbols = [Symbol(i) for i in range()]

    return symbols
    
def save_symbols(symbols, filename='symbols.pkl'):
    with open(filename, 'wb') as infile:
        pickle.dump(symbols, infile)


class Symbol():
    """ The symbol class which holds the keypoints and descriptors for a single symbol. """
    def __init__(self, id, kp=np.empty((0, 0)), desc=np.empty((0, 0))):
        self.id = id
        self.kp = kp
        self.desc = desc

