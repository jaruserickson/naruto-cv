"""
VidCtrl
"""

import cv2
import matplotlib.pyplot as plt
import threading
import time


class VidCtrl(threading.Thread):
    """ Video control object """
    def __init__(self, app, new_frame_event, end_frame_event, vidfile):
        threading.Thread.__init__(self)

        self._vidfile = vidfile
        self._cur_frame_num = 0
        self._next_frame_num = 1
        self._fps = 24

        self._app = app
        self._new_frame_event = new_frame_event
        self._end_frame_event = end_frame_event

        self._capture = cv2.VideoCapture(self._vidfile)
        self._cur_frame = None

    def load_frame(self):
        """ Load the next frame from file """
        ret = True
        frame = None

        while ret and self._cur_frame_num < self._next_frame_num:
            ret, frame = self._capture.read()
            self._cur_frame_num += 1

        self._next_frame_num += 1

        if ret:
            return frame
        else:
            return None

    def run(self):
        """ Run """
        if not self._capture.isOpened():
            print('Error opening video file')
            return 1

        # get the first frame
        self._cur_frame = self.load_frame()
        self._app.set_frame(self._cur_frame)
        
        timer = time.clock()

        # main loop
        while True:
            # wait for processing to finish
            self._end_frame_event.wait()
            output = self._app.get_output()
            
            if output is None:
                break

            # display the current frame
            print('Displaying frame')
            cv2.imshow('Application', self._cur_frame)
            
            time_left = max(1, int((1. / self._fps - (time.clock() - timer)) * 1000))
            if cv2.waitKey(time_left) & 0xFF == ord('q'):
                break
            else:
                timer = time.clock()

            # load the next frame and send it to the application
            self._cur_frame = self.load_frame()
            self._app.set_frame(self._cur_frame)

        # release 
        self._capture.release()
        cv2.destroyAllWindows()
        
        # signal end of video
        self._app.set_frame(None)

        return 0
