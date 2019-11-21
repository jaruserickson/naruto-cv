#!/bin/bash
mkdir ~/tensorflow
cd ~/tensorflow
git clone https://github.com/tensorflow/models.git
wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
mv faster_rcnn_inception_v2_coco_2018_01_28.tar.gz ~/tensorflow/models/research/object_detection/
cd ~/tensorflow/models/research/object_detection/
tar xf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz