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

    R(phi) is a 2-dimensional array whose rows are the (r, alpha)-values
    which indicate the possible locations of the symbol relative to an edge
    with gradient phi. The underlying R itself is thus a list of 2D arrays
    that holds these values for each phi.

    The format of R is such that any phi in [-pi, pi] is mapped to the values
    of both phi and phi + pi (or phi - pi if phi > 0). This is the purpose of
    the create_symbol function defined in init_symbols.py.
    """
    def __init__(self, id, dphi=None, R=None):
        self.id = id
        self._dphi = dphi
        self._R = R

    def R(self, phi):
        return self._R[int((np.pi + phi) / self._dphi)]


if __name__ == '__main__':
    symbols = load_symbols()
    print(symbols[0].kp)
    # save_symbols(symbols)
