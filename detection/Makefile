.DEFAULT_GOAL := all

all: install tf-migrate setup  # Requires sudo to run since it uses PIP.
tf-migrate: download migrate

# Need to set this to your $HOME.
HOME=/home/jacob
TF_FOLDER=$(HOME)/tensorflow1

install:
	@echo "Installing packages..."
	# PIP installations
	pip3 install protobuf pillow lxml cython contextlib2 jupyter matplotlib pandas opencv-python
	# Protoc Installation
	PROTOC_ZIP=protoc-3.7.1-linux-x86_64.zip
	curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v3.7.1/$$PROTOC_ZIP
	sudo unzip -o $$PROTOC_ZIP -d /usr/local bin/protoc
	sudo unzip -o $$PROTOC_ZIP -d /usr/local 'include/*'
	rm -f $$PROTOC_ZIP

setup:
	@echo "Setting up tensorflow/models..."
	# Protoc usage, create PB2 files
	cd $(TF_FOLDER)/models/research; \
		protoc --python_out=. ./object_detection/protos/anchor_generator.proto ./object_detection/protos/argmax_matcher.proto ./object_detection/protos/bipartite_matcher.proto ./object_detection/protos/box_coder.proto ./object_detection/protos/box_predictor.proto ./object_detection/protos/eval.proto ./object_detection/protos/faster_rcnn.proto ./object_detection/protos/faster_rcnn_box_coder.proto ./object_detection/protos/grid_anchor_generator.proto ./object_detection/protos/hyperparams.proto ./object_detection/protos/image_resizer.proto ./object_detection/protos/input_reader.proto ./object_detection/protos/losses.proto ./object_detection/protos/matcher.proto ./object_detection/protos/mean_stddev_box_coder.proto ./object_detection/protos/model.proto ./object_detection/protos/optimizer.proto ./object_detection/protos/pipeline.proto ./object_detection/protos/post_processing.proto ./object_detection/protos/preprocessor.proto ./object_detection/protos/region_similarity_calculator.proto ./object_detection/protos/square_box_coder.proto ./object_detection/protos/ssd.proto ./object_detection/protos/ssd_anchor_generator.proto ./object_detection/protos/string_int_label_map.proto ./object_detection/protos/train.proto ./object_detection/protos/keypoint_box_coder.proto ./object_detection/protos/multiscale_anchor_generator.proto ./object_detection/protos/graph_rewriter.proto ./object_detection/protos/calibration.proto ./object_detection/protos/flexible_grid_anchor_generator.proto
	# Install the tensorflow models research modules.	
	cd $(TF_FOLDER)/models/research; \
		python3 setup.py build; \
		python3 setup.py install

download:
	@ echo "Installing required modules..."
	# Download tensorflow/models and model of choice
	mkdir $(TF_FOLDER)
	cd $(TF_FOLDER); \
		git clone https://github.com/tensorflow/models.git; \
		wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz; \
		mv faster_rcnn_inception_v2_coco_2018_01_28.tar.gz $(TF_FOLDER)/models/research/object_detection/
	cd $(TF_FOLDER)/models/research/object_detection/; \
		tar xf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz; \
		rm faster_rcnn_inception_v2_coco_2018_01_28.tar.gz

migrate:
	@echo "Copying dataset"
	cd ../dataset; \
		cp -r images $(TF_FOLDER)/models/research/object_detection; \
		cp -r training $(TF_FOLDER)/models/research/object_detection; \
		cp train.record $(TF_FOLDER)/models/research/object_detection; \
		cp test.record $(TF_FOLDER)/models/research/object_detection


train:
	@echo "Training model on dataset..."
	# PATH setup
	export PYTHONPATH="/home/jacob/tensorflow/models:/home/jacob/tensorflow/models/research:/home/jacob/tensorflow/models/research/slim"; \
		export PATH="$$PATH:$$PYTHONPATH"; \
		cd $(TF_FOLDER)/models/research/object_detection; \
			cp $(TF_FOLDER)/models/research/object_detection/legacy/train.py $(TF_FOLDER)/models/research/object_detection;\
			python3 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/faster_rcnn_inception_v2_naruto.config

tensorboard:
	@echo "Spinning up tensorboard..."
	cd $(TF_FOLDER)/models/research/object_detection; \
		tensorboard --logdir=training

clean-data:
	@echo "Cleaning ~tensorflow/models/research/object_detection"
	rm -rf $(TF_FOLDER)/models/research/object_detection/images
	rm -rf $(TF_FOLDER)/models/research/object_detection/training
	rm $(TF_FOLDER)/models/research/object_detection/test.record
	rm $(TF_FOLDER)/models/research/object_detection/train.record

clean:
	rm -rf $(TF_FOLDER)