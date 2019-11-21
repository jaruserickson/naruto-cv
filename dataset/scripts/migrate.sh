#!/bin/bash
# Migrate stuff from here to ~/tensorflow

cp -r ../images ~/tensorflow/models/research/object_detection
cp -r ../training ~/tensorflow/models/research/object_detection
cp ../train.record ~/tensorflow/models/research/object_detection
cp ../test.record ~/tensorflow/models/research/object_detection