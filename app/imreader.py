"""
ImReader
"""

import cv2
import os
from os import path


class ImReader():
    """ Random access reader which reads images from a folder in depth first order. """
    def __init__(self, image_folder):
        self._root = image_folder
        self._image_list = []
        self._cur_index = -1
        # supported extensions gathered from opencv imread documentation
        self._supported_exts = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']

        temp_list = []

        if path.isdir(self._root):
            for fname in os.listdir(self._root):
                temp_list.append(path.join(self._root, fname))
        elif path.isfile(self._root):
            temp_list = [self._root]

        ind = 0

        # retrieve all image paths, recursing through directories (depth-first)
        while ind < len(temp_list):
            cur_path = path.join(temp_list[ind])

            if path.isfile(cur_path):
                    ext = path.splitext(cur_path)[1]

                    if ext in self._supported_exts:
                        self._image_list.append(cur_path)
            elif path.isdir(cur_path):
                back = temp_list[ind + 1:]
                temp_list = temp_list[:ind + 1] 

                for fname in os.listdir(cur_path):
                    temp_list.append(path.join(cur_path, fname))
                
                temp_list = temp_list + back

            ind += 1

    def __len__(self):
        return len(self._image_list)

    def start(self):
        if len(self._image_list) == 0:
            print('Cannot start image reader: no images at path specified')
            return 1
        else:
            return 0

    def stop(self):
        return

    def get_frame(self, offset=1):
        """ 
        Load the next image in the folder (depth-first), or the one at the offset if given.
        """
        self._cur_index =  min(len(self._image_list) - 1, max(0, self._cur_index + offset))
        return cv2.imread(self._image_list[self._cur_index])
