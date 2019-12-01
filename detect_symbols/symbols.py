"""
symbols.py

Anything that has to do with the symbols is in this file.
"""

import sys
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

SYMBOLS_DIR = os.path.dirname(os.path.realpath(__file__))
SYMBOLS_FILE = os.path.join(SYMBOLS_DIR, 'symbols.pkl')

# Constants defining range of symbol detection
# (if modified, must re-initialize symbols)
MIN_ROTATION = -45
MAX_ROTATION = 45
ROTATION_STEP = 5

MIN_SCALE = 10
MAX_SCALE = 200
SCALE_STEP = 10

MIN_GRAD = -180
MAX_GRAD = 180
GRAD_STEP = 10


def load_symbols(file=SYMBOLS_FILE):
    sys.path.insert(0, SYMBOLS_DIR)

    if os.path.isfile(file):
        with open(file, 'rb') as infile:
            symbols = pickle.load(infile)
    else:
        symbols = [Symbol(i) for i in range(len(SYMBOL_IDS))]

    sys.path.remove(SYMBOLS_DIR)
    return symbols
    
def save_symbols(symbols, file=SYMBOLS_FILE):
    sys.path.insert(0, SYMBOLS_DIR)

    with open(file, 'wb') as infile:
        pickle.dump(symbols, infile)

    sys.path.remove(SYMBOLS_DIR)


class Symbol():
    """ The symbol class which holds the R-table for a single symbol. 
    """
    def __init__(self, id, R=None, num_edges=None):
        self.id = id
        self._d_phi = GRAD_STEP
        self._R = R
        self.num_edges = num_edges

    def R(self, phi):
        return self._R[phi]


if __name__ == '__main__':
    symbols = load_symbols()
    print(symbols[0]._R)
    # save_symbols(symbols)
