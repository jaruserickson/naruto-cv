"""
application
"""

import cv2
import matplotlib.pyplot as plt
import time


class Application():
    def __init__(self):
        """ Initialize application """
        self._vidfile = None
        self._enable_character_recognition = None
        self._enable_village_symbol_detection = None
        self._start_frame = None
        self._stop_frame = None
        self._step_frame = None
        self._fps = 24

    """ Attribute setters """
    def set_vidfile(self, vidfile):
        self._vidfile = vidfile

    def set_start_frame(self, n):
        self._start_frame = n
        
    def set_stop_frame(self, n):
        self._stop_frame = n

    def set_step_frame(self, n):
        self._step_frame = n

    def set_enable_character_recognition(self, flag):
        self._enable_character_recognition = flag
        
    def set_enable_village_symbol_detection(self, flag):
        self._enable_village_symbol_detection = flag

    def run(self):
        """ Run the application """

        # open video file
        cap = cv2.VideoCapture(self._vidfile)

        if not cap.isOpened():
            print('Error opening video file')
            return 1

        n = 0
        timer = time.clock()

        # main loop
        while(True):
            # read frame
            ret, frame = cap.read()

            # check if valid frame
            if not ret:
                break

            if n < self._start_frame or n % self._step_frame != 0:
                continue

            if self._stop_frame > 0 and n >= self._stop_frame:
                break

            # do processing here

            # display
            cv2.imshow('Application', frame)
            
            time_left = max(1, int((1. / self._fps - (time.clock() - timer)) * 1000))
            if cv2.waitKey(time_left) & 0xFF == ord('q'):
                break
            else:
                timer = time.clock()

            n += 1

        # release 
        cap.release()
        cv2.destroyAllWindows()

        return 0
