"""
AlgoCtrl
"""

import threading
import time
from detect_symbols.symbol_detector import SymbolDetector
from detect_symbols import symbols


class AlgoCtrl(threading.Thread):
    """ Algo object that controls image processing """
    def __init__(self, app, in_frames, output, exit_event):
        threading.Thread.__init__(self)

        self._enable_character_recognition = None
        self._enable_village_symbol_detection = None

        self._app = app
        self._in_frames = in_frames
        self._output = output
        self._exit_event = exit_event

        self.verbose = False
        
    def set_enable_character_recognition(self, flag):
        self._enable_character_recognition = flag
        
    def set_enable_village_symbol_detection(self, flag):
        self._enable_village_symbol_detection = flag

    def run(self):
        """ Run """
        symbol_detector = SymbolDetector()

        # main loop
        while not self._exit_event.is_set():
            if not self._in_frames.empty() and not self._output.full():
                # get frame
                in_frame = self._in_frames.get()
                frame_num = in_frame['frame_id']
                frame = in_frame['frame']
                bnd_boxes = {}

                if frame is not None:
                    self.verbose and print(f'Algo: Processing frame {frame_num}')
                    
                    sym_id = 0

                    if 'img_name' in in_frame and in_frame['img_name'] is not None:
                        img_name = in_frame['img_name']
                        sym_id = int(img_name[0])

                    p, score = symbol_detector.process(frame, sym_id)

                    if p is not None:
                        bnd_boxes[symbols.SYMBOL_IDS[sym_id]] = p

                # send frame output to application
                output = {'frame_id': frame_num, 'frame': frame, 'bounding_boxes': bnd_boxes}
                self._output.put(output)
            else:
                # wait for new frame
                self.verbose and print('Algo: Waiting')
                time.sleep(0.01)

        self.verbose and print('AlgoCtrl: Quitting')
        return 0
