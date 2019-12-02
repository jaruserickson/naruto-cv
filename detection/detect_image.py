import argparse
import os
import sys
CWD = os.getcwd()
sys.path.extend([
    os.path.join(CWD, 'tensorflow/models/research/object_detection'),
    os.path.join(CWD, 'tensorflow/models/research/slim'),
    os.path.join(CWD, 'tensorflow/models/research'),
    os.path.join(CWD, 'tensorflow/models')])
import cv2
import tensorflow as tf
import numpy as np
from detect_video import BBox, filter_boxes, RetinaNetDetector, TensorFlowDetector, detect_yolo
# tensorflow utils
from utils import label_map_util
from utils import visualization_utils as vis_util
from retinanet.keras_retinanet.visualization import label_color


def main(img, detector, thresh=0.6, iou=0.2, display=True):
    """ main function. """
    # Download video
    sess = tf.compat.v1.Session()
    frame = cv2.imread(img)
    set_boxes = []
    boxes, scores, classes = detector.detect_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    # Non-maximum supression
    if len(boxes) > 0:
        indices = tf.image.non_max_suppression(boxes, scores, 20, iou_threshold=iou, score_threshold=thresh)
        with sess.as_default():
            indices = indices.eval()
            boxes = np.take(boxes, indices, 0)
            scores = np.take(scores, indices, 0)
            classes = np.take(classes, indices, 0)
            
        if isinstance(detector, TensorFlowDetector):
            box2color, box2name = filter_boxes(
                boxes,
                classes.astype(np.int32),
                scores,
                detector.get_class_index(),
                min_score_thresh=thresh)
                
            # Set boxes
            for box in box2color.keys():
                set_boxes.append(BBox(box, box2name[box], box2color[box]))
        elif isinstance(detector, RetinaNetDetector):
            for box, score, label in zip(boxes, scores, classes):
                color = tuple(label_color(label))
                label = f'{detector.label_to_name[int(label)]}: {score}'
                h, w = frame.shape[:2]
                xmin, ymin, xmax, ymax = box
                box = tuple([ymin / h, xmin / w, ymax / h, xmax / w])
                set_boxes.append(BBox(box, label, color))

        # Draw boxes
        for box in set_boxes:
            ymin, xmin, ymax, xmax = box.box
            vis_util.draw_bounding_box_on_image_array(
                frame,
                ymin, xmin, ymax, xmax,
                color=box.color,
                display_str_list=box.name if isinstance(detector, TensorFlowDetector) else [box.name],
                use_normalized_coordinates=True)

        cv2.imwrite(f"{img.split('.')[0]}_output.jpg", frame)
        if display:
            cv2.imshow('detector', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, default='image.jpg')
    parser.add_argument('--detector', type=str, default='frcnn')
    arg = parser.parse_args()
    img, detector = arg.image, arg.detector
    
    if detector == 'frcnn' or detector == 'ssd':
        print(f'Running {detector.upper()} Detector')
        main(img, detector=TensorFlowDetector(detector), thresh=0.6, display=False)
    elif detector == 'retina':
        print('Running RetinaNet Detector')
        main(img, detector=RetinaNetDetector(), thresh=0.6, iou=0.01, display=False)
    elif detector == 'yolo':
        print('Running YOLO Detector')
        detect_yolo(img, is_video=False)
    else:
        print('No valid detector specified.')
