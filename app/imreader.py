"""
ImReader
"""

import cv2
import os
from os import path
import threading
import time


class ImReader(threading.Thread):
    """ Random access reader which reads images from a folder in depth first order. """
    def __init__(self, vidctrl, image_folder, q, exit_event):
        threading.Thread.__init__(self)

        self._vidctrl = vidctrl
        self._root = image_folder
        self._image_list = []
        # supported extensions gathered from opencv imread documentation
        self._supported_exts = ['.bmp', '.dib', '.jpeg', '.jpg', '.jpe', '.jp2', '.png', '.webp', '.pbm', '.pgm', '.ppm', '.pxm', '.pnm', '.sr', '.ras', '.tiff', '.tif', '.exr', '.hdr', '.pic']
        self._q = q
        self._exit_event = exit_event

        # retrieve all image paths, recursing through directories (depth-first)
        temp_list = []

        if path.isdir(self._root):
            for fname in os.listdir(self._root):
                temp_list.append(path.join(self._root, fname))
        elif path.isfile(self._root):
            temp_list = [self._root]

        ind = 0

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
        
    def get_vid_info():
        return {
            'frame_range': (0, len(self._image_list))
        }

    def run(self):
        """ Run """
        if len(self._image_list) == 0:
            print('ImReader: No images at path specified')
            return 1

        while not self._exit_event.is_set():
            if not self._q.full():
                requested_index = self._vidctrl.get_next_frame_num()

                if requested_index < 0:
                    print('ImReader: Reached beginning of image set')
                    img = None
                elif requested_index >= len(self._image_list):
                    print('ImReader: Reached end of image set')
                    img = None
                else:
                    # read requested image
                    print(f'ImReader: Reading frame {requested_index}')
                    img = cv2.imread(self._image_list[requested_index])

                self._q.put({'frame_id': requested_index, 'frame': img})
            else:
                print('ImReader: Waiting')
                time.sleep(0.1)
        
        print('ImReader: Quitting')
        return 0
