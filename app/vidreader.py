"""
VidReader
"""

import cv2
import threading
import time


class VidReader(threading.Thread):
    def __init__(self, vidctrl, vidfile, q, exit_event):
        threading.Thread.__init__(self)

        self._vidctrl = vidctrl
        self._vidfile = vidfile
        self._capture = cv2.VideoCapture(vidfile)
        self._cur_frame_num = -1
        self._cur_frame = None
        self._q = q
        self._exit_event = exit_event

        self.verbose = False

    def run(self):
        """ Run """
        if not self._capture.isOpened():
            print('VidReader: Could not open video')
            return 1

        while not self._exit_event.is_set():
            if not self._q.full():
                requested_frame_num = self._vidctrl.get_next_frame_num()
                ret = True

                if requested_frame_num < self._cur_frame_num:
                    self._q.put({'frame_id': requested_frame_num, 'frame': None})

                # this reader can only read forward!
                while ret and self._cur_frame_num < requested_frame_num:
                    ret, self._cur_frame = self._capture.read()
                    self._cur_frame_num += 1

                if not ret:
                    self.verbose and print('VidReader: End of Video')
                    break

                self.verbose and print(f'VidReader: Reading frame {requested_frame_num}')
                self._q.put({'frame_id': requested_frame_num, 'frame': self._cur_frame})
            else:
                self.verbose and print('VidReader: Waiting')
                time.sleep(0.01)
        
        self.verbose and print('VidReader: Quitting')
        self._capture.release()
        return 0
