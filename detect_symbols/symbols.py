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
    """ The symbol class which holds the keypoints and descriptors for a single symbol. 
    
    Keypoints for the symbol are formatted differently than how they are output by SIFT,
    i.e. kp[i] = (r, s, phi, alpha) where,
        r - distance from keypoint to symbol mean
        s - scale of keypoint
        phi - orientation of keypoint
        alpha - angle of line going from keypoint to symbol mean
    """
    def __init__(self, id, kp=np.empty((0, 0)), desc=np.empty((0, 0))):
        self.id = id
        self.kp = kp
        self.desc = desc


if __name__ == '__main__':
    symbols = load_symbols()
    print(symbols[0].kp)
    save_symbols(symbols)
