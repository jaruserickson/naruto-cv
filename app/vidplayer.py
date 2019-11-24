"""
VidPlayer
"""

import cv2
import threading
import time


class VidPlayer(threading.Thread):
    def __init__(self, vidctrl, name, q, exit_event, mode='video', fps=24):
        threading.Thread.__init__(self)

        self._vidctrl = vidctrl
        #self._vid_info = vid_info
        self._q = q
        self._exit_event = exit_event
        
        self._window_name = name
        self._cur_frame_num = -1 
        self._next_frame_num = 0
        self._last_requested_frame_num = -1
        self._mode = mode
        self._fps = fps

        if self._mode == 'images':
            self._fps = 0
        
    def process_key_event(self, key):
        """ Process a key event """
        if key & 0xFF == ord('q'):
            return 1
        else:
            # video mode
            if self._mode == 'video':
                if key & 0xFF == ord(' '):
                    while cv2.waitKey(0) & 0xFF != ord(' '):
                        pass
                elif key & 0xFF == ord('n'):
                    self._next_frame_num = self._cur_frame_num + 24 * 10
            # image mode
            elif self._mode == 'images':
                if key & 0xFF == ord('p'):
                    self._next_frame_num = self._cur_frame_num - 1
                elif key != -1:
                    self._next_frame_num = self._cur_frame_num + 1
            
        return 0

    def run(self):
        """ Run """
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_EXPANDED)
        timer = time.clock()

        self._vidctrl.request_frame_num(self._next_frame_num)
        self._last_requested_frame_num = self._next_frame_num

        while not self._exit_event.is_set():
            if not self._q.empty():
                output = self._q.get()

                if output['frame_id'] == self._next_frame_num:
                    if output['frame'] is None:
                        # invalid frame
                        print(f"VidPlayer: Invalid frame {output['frame_id']} - Retrying")
                        self._vidctrl.request_frame_num(self._next_frame_num)
                        wait_time = 1
                    else:
                        # display the current frame
                        print(f'VidPlayer: Displaying frame {self._next_frame_num}')
                        cv2.imshow(self._window_name, output['frame'])

                        self._cur_frame_num = self._next_frame_num
                        self._next_frame_num += 1

                        if self._fps == 0:
                            wait_time = 0
                        else:
                            wait_time = max(1, int((1. / self._fps - (time.clock() - timer)) * 1000))
                else:
                    print(f"VidPlayer: Discarding frame {output['frame_id']}")
                    wait_time = 1
            else:
                print('VidPlayer: Waiting')
                wait_time = 10

            # wait and process key events
            key = cv2.waitKey(wait_time)
            timer = time.clock()
            ret = self.process_key_event(key)
            
            if ret != 0:
                print('VidPlayer: Quit requested')
                break

            if self._last_requested_frame_num != self._next_frame_num:
                self._vidctrl.request_frame_num(self._next_frame_num)
                self._last_requested_frame_num = self._next_frame_num
        
        print('VidPlayer: Quitting')
        cv2.destroyWindow(self._window_name)
        return 0
