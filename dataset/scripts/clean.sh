#!/bin/bash
# Clean up ~/tensorflow

rm -rf ~/tensorflow/models/research/object_detection/images
rm -rf ~/tensorflow/models/research/object_detection/training
rm ~/tensorflow/models/research/object_detection/test.record
rm ~/tensorflow/models/research/object_detection/train.record