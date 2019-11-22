""" Run our detection/tracking software on the video specified. """
import sys
import os
from collections import defaultdict
CWD = os.getcwd()
sys.path.extend([
    os.path.join(CWD, 'tensorflow/models/research/object_detection'),
    os.path.join(CWD, 'tensorflow/models/research/slim'),
    os.path.join(CWD, 'tensorflow/models/research'),
    os.path.join(CWD, 'tensorflow/models')])
import cv2
from pytube import YouTube
import tensorflow as tf
import numpy as np
# tensorflow utils
from utils import label_map_util
from utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

class Detector():
    """ Create detector based on training. """
    def __init__(self):
        """ Return bounding boxes for an image.
        Modified from @EdjeElectronics on github:
        https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
        """
        print('Creating detector based on frozen graph...')
        graph_path = os.path.join(os.getcwd(), \
            'tensorflow/models/research/object_detection/inference_graph', 'frozen_inference_graph.pb')
        label_path = os.path.join(os.getcwd(), \
            'tensorflow/models/research/object_detection/training', 'labelmap.pbtxt')
        num_classes = 27

        label_map = label_map_util.load_labelmap(label_path)
        classes = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
        self.class_index = label_map_util.create_category_index(classes)
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(graph_path, 'rb') as fid:
                serialzd_graph = fid.read()
                od_graph_def.ParseFromString(serialzd_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.compat.v1.Session(graph=detection_graph) 

        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self.fetches = [detection_boxes, detection_scores, detection_classes, num_detections]

    def detect_image(self, image):
        """ Run the actual detector """
        image_xp = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run(self.fetches,
            feed_dict={self.image_tensor: image_xp})
        return (boxes, scores, classes, num)

    def get_class_index(self):
        """ Getter for class index """
        return self.class_index


def download_video(tag='ns_op18'):
    """ Download a chosen video for use. """
    videos = {
        'ns_op18': 'https://www.youtube.com/watch?v=HdgD7E6JEE4',
        'n_op2': 'https://www.youtube.com/watch?v=SRn99oN1p_c',
        'naruto_hinata_wedding': 'https://www.youtube.com/watch?v=BoMBsDIGkKI'}

    return YouTube(videos[tag]).streams.first().download()

def filter_boxes(boxes, classes, scores, class_index, min_score_thresh):
    """ Modified version of tensorflow's visualize_boxes_and_labels_on_image_array """
    box_to_display_str_map = defaultdict(list)
    box_to_color_map = defaultdict(str)
    max_boxes_to_draw = 20
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                display_str = ''
                if classes[i] in class_index.keys():
                    class_name = class_index[classes[i]]['name']
                else:
                    class_name = 'N/A'
                display_str = str(class_name)
                if not display_str:
                    display_str = '{}%'.format(int(100 * scores[i]))
                else:
                    display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                box_to_display_str_map[box].append(display_str)
                box_to_color_map[box] = STANDARD_COLORS[classes[i] % len(STANDARD_COLORS)]

    return box_to_color_map, box_to_display_str_map


if __name__ == '__main__':
    # Download video
    vid_path = download_video()
    cap = cv2.VideoCapture(vid_path)
    multi_tracker = cv2.MultiTracker_create()
    det = Detector()
    while cap.isOpened():
        success, frame = cap.read()
        boxes, scores, classes, num = det.detect_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Get boxes
        box2color, box2name = filter_boxes(
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            det.get_class_index(),
            min_score_thresh=0.60)
        # Draw box
        for box in box2color.keys():
            ymin, xmin, ymax, xmax = box
            vis_util.draw_bounding_box_on_image_array(
                frame,
                ymin, xmin, ymax, xmax,
                color=box2color[box],
                display_str_list=box2name[box],
                use_normalized_coordinates=True)

        # Tracker logic
        # if len(box2color) > 0:
        #     for bbox in box2color.keys():
        #         h, w = frame.shape[:2]
        #         
        #         bbox = (round(ymin * h), round(xmin * w), round(ymax * h), round(xmax * w))
        #         multi_tracker.add(cv2.TrackerCSRT_create(), frame, bbox)

        # success, boxes = multi_tracker.update(frame)
        # for i, newbox in enumerate(boxes):
        #     vis_util.draw_bounding_box_on_image_array(frame, **newbox, color=COLORS[i % len(COLORS)] use_normalized_coordinates=False)
        
        cv2.imshow('detector', frame)
        cv2.waitKey(1)
        

    os.remove(vid_path)