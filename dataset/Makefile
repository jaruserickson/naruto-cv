.DEFAULT_GOAL := all

all: mal google-images youtube dataset

mal:
	@echo "Downloading images from MyAnimeList..."
	python3 download_mal.py

google-images:
	@echo "Downloading validated Google Images..."
	python3 download_validated.py

youtube:
	@echo "Downloading validated YouTube frames..."
	python3 youtube_frame_validator.py 0
	python3 youtube_frame_extractor.py

dataset:
	@echo "Generating dataset files..."
	python3 generate_split.py
	python3 generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
	python3 generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

clean:
	@echo "Cleaning dataset..."
	rm *.record
	rm google_images.txt
	rm -rf vid_data/frames vid_data/videos data images