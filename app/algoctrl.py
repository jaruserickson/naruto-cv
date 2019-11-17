"""
AlgoCtrl
"""

import threading
import time


class AlgoCtrl(threading.Thread):
    """ Algo object that controls image processing """
    def __init__(self, app, new_frame_event, end_frame_event):
        threading.Thread.__init__(self)

        self._enable_character_recognition = None
        self._enable_village_symbol_detection = None

        self._app = app
        self._new_frame_event = new_frame_event
        self._end_frame_event = end_frame_event
        
    def set_enable_character_recognition(self, flag):
        self._enable_character_recognition = flag
        
    def set_enable_village_symbol_detection(self, flag):
        self._enable_village_symbol_detection = flag

    def run(self):
        """ Run """
        # main loop
        while True:
            # wait foor a new frame
            self._new_frame_event.wait()
            frame = self._app.get_frame()

            if frame is None:
                break

            print('Processing frame')

            # send frame output to application
            self._app.set_output([1])

        # signal end of processing
        self._app.set_output(None)
        return 0
