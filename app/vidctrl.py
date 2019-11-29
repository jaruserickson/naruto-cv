"""
VidCtrl
"""

import cv2
import matplotlib.pyplot as plt
import threading
import time
import queue

from .vidreader import VidReader
from .imreader import ImReader
from .vidplayer import VidPlayer


class VidCtrl(threading.Thread):
    """ 
    Video control object 
    
    Rather than simply playing a video (as would a video player), VidCtrl also 
    controls the sending of frames to the main application.
    """
    def __init__(self, app, in_frames, output, vidfile, exit_event):
        threading.Thread.__init__(self)

        self._app = app
        self._in_frames = in_frames
        self._output = output
        self._vidfile = vidfile
        self._exit_event = exit_event

        self._requested_frame_nums = queue.Queue()
        self._unrequested_frame_nums = queue.Queue()
        self._last_read_frame_num = -1

        self.verbose = False

    def set_mode(self, mode):
        self._mode = mode

    def set_fps(self, fps):
        self._fps = fps

    def get_next_frame_num(self):
        if self._requested_frame_nums.empty():
            # if the video player has not requested a frame, get the frame after the last one
            self._last_read_frame_num += 1
            self._unrequested_frame_nums.put(self._last_read_frame_num)
            return self._last_read_frame_num
        else:
            requested_frame_num = self._requested_frame_nums.get()
            is_read = False

            # check to see if this number has been pre-read
            while not self._unrequested_frame_nums.empty():
                if requested_frame_num == self._unrequested_frame_nums.get():
                    is_read = True
                    break
            
            if is_read:
                return self.get_next_frame_num()
            else:
                self._last_read_frame_num = requested_frame_num
                return requested_frame_num

    def request_frame_num(self, n):
        self._requested_frame_nums.put(n)

    def run(self):
        """ Run """

        child_exit_event = threading.Event()

        # create video provider
        if self._mode == 'video':
            vid_reader = VidReader(self, self._vidfile, self._in_frames, child_exit_event)
        elif self._mode == 'images':
            vid_reader = ImReader(self, self._vidfile, self._in_frames, child_exit_event)
        else:
            print('Unknown mode: quitting VidCtrl')
            return 1

        # create video player
        vid_player = VidPlayer(self, 'Naruto-CV', self._output, child_exit_event, self._mode, self._fps)

        # run
        vid_reader.start()
        vid_player.start()

        # wait for termination
        while vid_reader.is_alive() and vid_player.is_alive() and not self._exit_event.is_set():
            time.sleep(0.1)

        child_exit_event.set()
        vid_reader.join()
        vid_player.join()

        self.verbose and print('VidCtrl: Quitting')
        return 0
