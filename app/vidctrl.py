"""
VidCtrl
"""

import cv2
import matplotlib.pyplot as plt
import threading
import time

from vidreader import VidReader
from imreader import ImReader


class VidCtrl(threading.Thread):
    """ Video control object """
    def __init__(self, app, new_frame_event, end_frame_event, vidfile):
        threading.Thread.__init__(self)

        self._vidfile = vidfile
        self._frame_num = 0
        self._frame_offset = 1
        self._mode = None
        self._fps = None

        self._app = app
        self._new_frame_event = new_frame_event
        self._end_frame_event = end_frame_event

        self._vid_reader = None
        self._window_name = 'Naruto CV'
        self._cur_frame = None

    def set_mode(self, mode):
        self._mode = mode

    def set_fps(self, fps):
        self._fps = fps

    def key_event(self, key):
        """ Process a key event """
        # Note: In 'video' mode, '_next_frame_num' is incremented by 1 every frame in the main loop
        if key & 0xFF == ord('q'):
            return 1
        else:
            # video mode
            if self._mode == 'video':
                if key & 0xFF == ord(' '):
                    while cv2.waitKey(0) & 0xFF != ord(' '):
                        pass
                elif key & 0xFF == ord('n'):
                    self._frame_offset = self._fps * 10
            # image mode
            elif self._mode == 'images':
                if key & 0xFF == ord('p'):
                    self._frame_offset = -1
                else:
                    self._frame_offset = 1
            
        return 0

    def run(self):
        """ Run """
        # safety checks
        if self._mode is None or self._fps is None:
            print('VidCtrl not initialized')
            return 1

        # create video provider
        if self._mode == 'video':
            self._vid_reader = VidReader(self._vidfile)
        elif self._mode == 'images':
            self._vid_reader = ImReader(self._vidfile)
            self._fps = 0
        else:
            print('Unknown mode: quitting VidCtrl')
            return 1
            
        # more safety checks
        if self._vid_reader.start() != 0:
            print('Error opening video file')
            return 1

        # get the first frame
        self._cur_frame = self._vid_reader.get_frame(self._frame_offset)
        self._app.set_frame(self._cur_frame)
        self._frame_num = self._frame_offset
        
        # initialize window
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        timer = time.clock()

        # main loop
        while self._cur_frame is not None:
            # wait for processing to finish
            self._end_frame_event.wait()
            output = self._app.get_output()
            
            if output is None:
                break

            # display the current frame
            print('Displaying frame')
            cv2.imshow(self._window_name, self._cur_frame)
            
            # wait and process key events
            if self._fps == 0:
                wait_time = 0
            else:
                wait_time = max(1, int((1. / self._fps - (time.clock() - timer)) * 1000))

            key = cv2.waitKey(wait_time)

            if self.key_event(key) != 0:
                break
            else:
                timer = time.clock()

            # load the next frame and send it to the application
            self._cur_frame = self._vid_reader.get_frame(self._frame_offset)
            self._app.set_frame(self._cur_frame)

            if self._mode == 'video':
                self._frame_num += self._frame_offset
                self._frame_offset = 1

        # release 
        self._vid_reader.stop()
        cv2.destroyAllWindows()
        
        # signal end of video
        self._app.set_frame(None)

        return 0
