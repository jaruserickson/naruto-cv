import time
import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras_retinanet.visualization import draw_box, draw_caption, label_color
from keras_retinanet.util import UpsampleLike, RegressBoxes, FilterDetections, PriorProbability, ClipBoxes
from keras_retinanet.losses import smooth_l1, focal
from keras_retinanet.anchors import Anchors
from keras_retinanet.image import read_image_bgr, preprocess_image, resize_image
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = keras.models.load_model('out.h5', custom_objects={
    'UpsampleLike'     : UpsampleLike,
    'PriorProbability' : PriorProbability,
    'RegressBoxes'     : RegressBoxes,
    'FilterDetections' : FilterDetections,
    'Anchors'          : Anchors,
    'ClipBoxes'        : ClipBoxes,
    '_smooth_l1'       : smooth_l1(),
    '_focal'           : focal(),
}, compile=False)

label_to_name = [
    'naruto_uzumaki=',
    'sasuke_uchiha',
    'kakashi_hatake',
    'gaara',
    'itachi_uchiha',
    'deidara',
    'minato_namikaze',
    'shikamaru_nara',
    'hinata_hyuuga',
    'sakura_haruno',
    'sai',
    'yamato',
    'neji_hyuuga',
    'jiraya',
    'temari',
    'rock_lee',
    'kushina_uzumaki',
    'kisame_hoshigaki',
    'killer_bee',
    'might_guy',
    'kiba_inuzuka',
    'ino_yamanaka',
    'sasori',
    'pain',
    'konan',
    'iruka_umino',
    'shino_aburame']

images = os.listdir('../../dataset/retimages/train')
np.random.shuffle(images)
for filepath in images:
    if filepath.split('.')[-1] == 'png':
        path = os.path.abspath(os.path.join('../../dataset/retimages/train', filepath))
        image = read_image_bgr(path)
    else:
        continue

    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    image = preprocess_image(image)
    image, scale = resize_image(image)

    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    print("processing time: ", time.time() - start)

    iou_threshold = 0.3

    boxes /= scale
    boxes, scores, labels = boxes[0], scores[0], labels[0]
    if len(boxes[0]) > 0:
        indices = tf.image.non_max_suppression(boxes, scores, 20, iou_threshold=iou_threshold, score_threshold=0.5)
        sess = tf.compat.v1.Session()
        with sess.as_default():
            indices = indices.eval()
            boxes = np.take(boxes, indices, 0)
            scores = np.take(scores, indices, 0)
            labels = np.take(labels, indices, 0)

    for box, score, label in zip(boxes, scores, labels):
        color = label_color(label)
        b = box.astype(int)
        draw_box(draw, b, color=color)
        caption = "{} {:.3f}".format(label_to_name[int(label)], score)
        draw_caption(draw, b, caption)
        
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    plt.imshow(draw)
    plt.show()
