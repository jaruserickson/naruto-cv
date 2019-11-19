"""
application
"""

import cv2
import time
import threading

from algoctrl import AlgoCtrl
from vidctrl import VidCtrl


class Application():
    """ Main application """
    def __init__(self):
        """ Initialize application """
        self._args = None

        self._output = None
        self._frame = None
        self._frame_lock = threading.Lock()
        self._output_lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._end_frame_event = threading.Event()

    """ Attribute setters """
    def set_args(self, args):
        self._args = args

    def run(self):
        """ Run the application """
        algo = AlgoCtrl(self, self._new_frame_event, self._end_frame_event)
        algo.set_enable_character_recognition(self._args.enable_character_recognition)
        algo.set_enable_village_symbol_detection(self._args.enable_village_symbol_detection)

        video = VidCtrl(self, self._new_frame_event, self._end_frame_event, self._args.vid_file)
        video.set_mode(self._args.mode)
        video.set_fps(self._args.fps)

        video.start()
        algo.start()

        video.join()
        algo.join()

        return 0

    def set_frame(self, frame):
        """ Set the next frame for processing """
        self._frame_lock.acquire()
        self._frame = frame
        self._frame_lock.release()

        # trigger new frame event
        self._new_frame_event.set()

    def get_frame(self):
        """ Retrieve the current frame """
        self._frame_lock.acquire()

        if self._frame is not None:
            frame = self._frame.copy()
        else:
            frame = None

        self._frame_lock.release()
        self._new_frame_event.clear()
        return frame

    def set_output(self, output):
        """ Set the output of the most recent frame """
        self._output_lock.acquire()
        self._output = output
        self._output_lock.release()

        # trigger end of frame
        self._end_frame_event.set()

    def get_output(self):
        """ Get the most recent frame's output """
        self._output_lock.acquire()

        if self._output is not None:
            output = self._output.copy()
        else:
            output = None

        self._output_lock.release()
        self._end_frame_event.clear()
        return output
