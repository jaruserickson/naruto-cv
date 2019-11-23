#!/bin/bash
# 1: latest checkpoint
export PYTHONPATH="${PWD}/tensorflow/models:${PWD}/tensorflow/models/research:${PWD}/tensorflow/models/research/slim:${PWD}/tensorflow/models/research/object_detection"
export PATH="$PATH:$PYTHONPATH"
OBJ=./tensorflow/models/research/object_detection
cd ${OBJ} && \
	rm -rf inference_graph/; \
	python3 export_inference_graph.py \
		--input_type image_tensor \
		--pipeline_config_path training/faster_rcnn_inception_v2_naruto.config \
		--trained_checkpoint_prefix training/model.ckpt-${1} \
		--output_directory inference_graph
