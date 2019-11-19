"""
VidReader
"""

import cv2


class VidReader():
    def __init__(self, vidfile):
        self._vidfile = vidfile
        self._capture = cv2.VideoCapture(vidfile)
        self._cur_frame_num = -1
        self._cur_frame = None

    def start(self):
        if self._capture.isOpened():
            return 0
        else:
            return 1

    def stop(self):
        self._capture.release()
        return

    def get_frame(self, offset=1):
        """ 
        Load the next frame of the video, or the one at the offset if given.
        
        If offset is less than zero, will simply return the last frame.
        """
        ret = True
        i = 0

        while ret and i < offset:
            ret, self._cur_frame = self._capture.read()
            i += 1

        self._cur_frame_num += offset

        if ret:
            return self._cur_frame
        else:
            return None
