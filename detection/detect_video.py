""" Run our detection/tracking software on the video specified. """
import sys
import os
import subprocess
import time
import argparse
from collections import defaultdict
CWD = os.getcwd()
sys.path.extend([
    os.path.join(CWD, 'tensorflow/models/research/object_detection'),
    os.path.join(CWD, 'tensorflow/models/research/slim'),
    os.path.join(CWD, 'tensorflow/models/research'),
    os.path.join(CWD, 'tensorflow/models')])
import cv2
from tqdm import tqdm
from pytube import YouTube
import tensorflow as tf
from tensorflow import keras
import numpy as np
# tensorflow utils
from utils import label_map_util
from utils import visualization_utils as vis_util
# RetinaNet
from retinanet.keras_retinanet.visualization import draw_box, draw_caption, label_color
from retinanet.keras_retinanet.util import UpsampleLike, RegressBoxes, FilterDetections, PriorProbability, ClipBoxes
from retinanet.keras_retinanet.losses import smooth_l1, focal
from retinanet.keras_retinanet.anchors import Anchors
from retinanet.keras_retinanet.image import preprocess_image, resize_image
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
VIDEOS = {
    'ns_op18': {
        'name': 'Naruto Shippuden Opening 18  LINE (HD).mp4',
        'uri': 'https://www.youtube.com/watch?v=HdgD7E6JEE4'},
    'n_op2': {
        'name': 'Naruto Opening 2  Haruka Kanata (HD).mp4',
        'uri': 'https://www.youtube.com/watch?v=SRn99oN1p_c'},
    'naruto_hinata_wedding': {
        'name': 'Naruto and Hinata Wedding.mp4',
        'uri': 'https://www.youtube.com/watch?v=BoMBsDIGkKI'},
    'naruto_v_sasuke': {
        'name': '【MAD】 Naruto VS Sasuke  ナルト VS サスケ 『アウトサイダー』.mp4',
        'uri': 'https://www.youtube.com/watch?v=u_1onhckHuw'},
    'naruto_chill': {
        'name': "Kakashi's mask",
        'uri': 'https://www.youtube.com/watch?v=UGn-Tg1j8w0'},
    'sasuke_oroch': {
        'name': 'Sasuke vs Orochimaru  Naruto.mp4',
        'uri': 'https://www.youtube.com/watch?v=MwJUK2JtSgw'}}

class RetinaNetDetector():
    """ Create RetinaNet Detector based on trianing. """
    def __init__(self):
        """ Return bounding boxes for an image.
        Modified from @EdjeElectronics on github:
        https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
        """
        print('Creating detector based on snapshots...')
        
        self.model = keras.models.load_model('../frozen_weights/retina.h5', custom_objects={
            'UpsampleLike'     : UpsampleLike,
            'PriorProbability' : PriorProbability,
            'RegressBoxes'     : RegressBoxes,
            'FilterDetections' : FilterDetections,
            'Anchors'          : Anchors,
            'ClipBoxes'        : ClipBoxes,
            '_smooth_l1'       : smooth_l1(),
            '_focal'           : focal(),
        }, compile=False)

        self.label_to_name = [
            'naruto_uzumaki', 'sasuke_uchiha', 'kakashi_hatake', 'gaara', \
                'itachi_uchiha', 'deidara', 'minato_namikaze', 'shikamaru_nara', \
                    'hinata_hyuuga', 'sakura_haruno', 'sai', 'yamato', 'neji_hyuuga', \
                        'jiraya', 'temari', 'rock_lee', 'kushina_uzumaki', 'kisame_hoshigaki', \
                            'killer_bee', 'might_guy', 'kiba_inuzuka', 'ino_yamanaka', 'sasori', \
                                'pain', 'konan', 'iruka_umino', 'shino_aburame']
        
    def detect_image(self, image):
        """ Run the actual detector """
        image = preprocess_image(image)
        image, scale = resize_image(image)

        start = time.time()
        boxes, scores, labels = self.model.predict_on_batch(np.expand_dims(image, axis=0))
        iou_threshold = 0.3
        score_threshold = 0.3

        boxes /= scale
        return (boxes[0], scores[0], labels[0])

class TensorFlowDetector():
    """ Create Faster R-CNN detector based on training. """
    def __init__(self, model='frcnn'):
        """ Return bounding boxes for an image.
        Modified from @EdjeElectronics on github:
        https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10
        """
        print('Creating detector based on frozen graph...')
        graph_path = os.path.join('../frozen_weights', 'frcnn.pb' if model=='frcnn' else 'ssd.pb')
        label_path = os.path.join('tensorflow/models/research/object_detection/training', 'labelmap.pbtxt')
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
        (boxes, scores, classes, _) = self.sess.run(self.fetches,
            feed_dict={self.image_tensor: image_xp})
        return (boxes[0], scores[0], classes[0])

    def get_class_index(self):
        """ Getter for class index """
        return self.class_index

def detect_yolo(vid_choice):
    """ Detect a video using weights from YOLO model. """
    vid_path = download_video(vid_choice)
    try:
        # subprocess.check_call(f'python3 yolov3/detect.py --weights ../frozen_weights/yolo_last.pt --source "{vid_path}" --data yolo.data --cfg {os.path.abspath("../dataset/yolov3.cfg")}', shell=True)
        # Attatch audio
        inname = VIDEOS[vid_choice]['name']
        outname = os.path.join('output', VIDEOS[vid_choice]['name'])
        subprocess.call(f'ffmpeg -i "{inname}" -ab 160k -ac 2 -ar 44100 -vn {vid_choice}_audio.wav', shell=True)
        subprocess.call(f'ffmpeg -i "{outname}" -i {vid_choice}_audio.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {vid_choice}_audio.mp4', shell=True)
    except subprocess.CalledProcessError as e:
        print(f'Failure: {e}')
    
class BBox():
    """ Bounding box object """
    def __init__(self, box, name, color):
        self.box = box
        self.name = name
        self.color = color
    
    def __str__(self):
        return f'Bounding Box:\n\tName: {self.name}\n\tColor: {self.color}\n\tBounds: {self.box}'

def download_video(tag='ns_op18'):
    """ Download a chosen video for use. """
    if VIDEOS[tag]['name'] not in os.listdir(os.getcwd()):
        return YouTube(VIDEOS[tag]['uri']).streams.first().download()
    else:
        return os.path.join(os.getcwd(), VIDEOS[tag]['name'])

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

def create_tracker(frame, boxes):
    """ Create a tracker with the boxes and the frame provided. """
    h, w = frame.shape[:2]
    tracker = cv2.MultiTracker_create()
    for box in boxes:
        ymin, xmin, ymax, xmax = box.box
        bbox = (round(ymin * h), round(xmin * w), round(ymax * h), round(xmax * w))
        tracker.add(cv2.TrackerMedianFlow_create(), frame, bbox)

    return tracker

def main(vid_choice, detector, detect_rate=5, thresh=0.6, display=True):
    """ main function. """
    # Download video
    vid_path = download_video(vid_choice)
    cap = cv2.VideoCapture(vid_path)
    multi_tracker = cv2.MultiTracker_create()
    det = detector
    out = cv2.VideoWriter(f'{vid_choice}.mp4', \
        cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), \
            (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    progress = tqdm(total=cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sess = tf.compat.v1.Session()
    iou_threshold = 0.2
    score_threshold = thresh
    count = 0
    tracking_boxes = []
    while cap.isOpened():
        progress.update(1)
        success, frame = cap.read()
        if not success:
            print('Video read failed...')
            break
        # Run a new detection:
        if count % detect_rate == 0:
            # Get detections
            boxes, scores, classes = det.detect_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            # Non-maximum supression
            if len(boxes) > 0:
                indices = tf.image.non_max_suppression(boxes, scores, 20, iou_threshold=iou_threshold, score_threshold=score_threshold)
                with sess.as_default():
                    indices = indices.eval()
                    boxes = np.take(boxes, indices, 0)
                    scores = np.take(scores, indices, 0)
                    classes = np.take(classes, indices, 0)
            
            tracking_boxes = []
            if isinstance(det, TensorFlowDetector):
                box2color, box2name = filter_boxes(
                    boxes,
                    classes.astype(np.int32),
                    scores,
                    det.get_class_index(),
                    min_score_thresh=thresh)
                
                # Get boxes, replace tracking boxes
                for box in box2color.keys():
                    tracking_boxes.append(BBox(box, box2name[box], box2color[box]))
            elif isinstance(det, RetinaNetDetector):
                for box, score, label in zip(boxes, scores, classes):
                    color = tuple(label_color(label))
                    label = f'{det.label_to_name[int(label)]}: {score}'
                    h, w = frame.shape[:2]
                    ymin, xmin, ymax, xmax = box
                    box = tuple([ymin / h, xmin / w, ymax / h, xmax / w])
                    tracking_boxes.append(BBox(box, label, color))

            # Set the tracker.
            multi_tracker = create_tracker(frame, tracking_boxes)
        else: # Use a tracker for the in-between frames
            # Get boxes from the tracker.
            success, boxes = multi_tracker.update(frame)
            meta = [(b.name, b.color) for b in tracking_boxes]
            tracking_boxes = []
            for i, newbox in enumerate(boxes):
                h, w = frame.shape[:2]
                ymin, xmin, ymax, xmax = newbox
                newbox = (ymin / h, xmin / w, ymax / h, xmax / w)
                tracking_boxes.append(BBox(newbox, meta[i][0], meta[i][1]))

        # Draw boxes
        for box in tracking_boxes:
            ymin, xmin, ymax, xmax = box.box
            vis_util.draw_bounding_box_on_image_array(
                frame,
                ymin, xmin, ymax, xmax,
                color=box.color,
                display_str_list=box.name if isinstance(det, TensorFlowDetector) else [box.name],
                use_normalized_coordinates=True)

        out.write(frame)
        if display:
            cv2.imshow('detector', frame)
            cv2.waitKey(1)
        count += 1

    cap.release()
    out.release()

    # Attach audio.
    subprocess.call(f'ffmpeg -i "{vid_path}" -ab 160k -ac 2 -ar 44100 -vn {vid_choice}_audio.wav', shell=True)
    subprocess.call(f'ffmpeg -i "{vid_choice}.mp4" -i {vid_choice}_audio.wav -c:v copy -c:a aac -strict experimental -map 0:v:0 -map 1:a:0 {vid_choice}_audio.mp4', shell=True)

    # Clean out temporary files.
    # os.remove(vid_path)
    # os.remove(f'{vid_choice}_audio.wav')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('video', type=str, default='naruto_v_sasuke')
    parser.add_argument('--detector', type=str, default='frcnn')
    arg = parser.parse_args()
    vid, detector = arg.video, arg.detector
    
    if detector == 'frcnn' or detector == 'ssd':
        print(f'Running {detector.upper()} Detector')
        main(vid, detector=TensorFlowDetector(detector), detect_rate=2, thresh=0.6, display=False)
    elif detector == 'retina':
        print('Running RetinaNet Detector')
        main(vid, detector=RetinaNetDetector(), detect_rate=2, thresh=0.6, display=False)
    elif detector == 'yolo':
        print('Running YOLO Detector')
        detect_yolo(vid)
    else:
        print('No valid detector specified.')
