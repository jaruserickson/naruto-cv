"""
application
"""

import cv2
import queue
import time
import threading

from .algoctrl import AlgoCtrl
from .vidctrl import VidCtrl


class Application():
    """ Main application """
    def __init__(self):
        """ Initialize application """
        self._args = None
        self._buf_size = 10

    """ Attribute setters """
    def set_args(self, args):
        self._args = args

    def run(self):
        """ Run the application """
        in_frames = queue.Queue(self._buf_size)
        output = queue.Queue(self._buf_size)

        exit_event = threading.Event()

        # create algo and video control objects
        algo = AlgoCtrl(self, in_frames, output, exit_event)
        algo.set_enable_character_recognition(self._args.enable_character_recognition)
        algo.set_enable_village_symbol_detection(self._args.enable_village_symbol_detection)

        video = VidCtrl(self, in_frames, output, self._args.vid_file, exit_event)
        video.set_mode(self._args.mode)
        video.set_fps(self._args.fps)

        # run
        video.start()
        algo.start()

        # wait for termination
        while video.is_alive() and algo.is_alive():
            time.sleep(0.1)

        exit_event.set()
        video.join()
        algo.join()
        
        return 0
