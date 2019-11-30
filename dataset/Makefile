.DEFAULT_GOAL := all

all: download youtube dataset
download: mal google-images

mal:
	@echo "Downloading images from MyAnimeList..."
	python download_mal.py

google-images:
	@echo "Downloading validated Google Images..."
	python download_validated.py

youtube:
	@echo "Downloading validated YouTube frames..."
	python youtube_frame_validator.py 0
	python youtube_frame_extractor.py

dataset:
	@echo "Generating dataset files..."
	python generate_split.py
	python generate_tfrecord.py --csv_input=images/train_labels.csv --image_dir=images/train --output_path=train.record
	python generate_tfrecord.py --csv_input=images/test_labels.csv --image_dir=images/test --output_path=test.record

fix_png:
	@echo "Fixing pngs in images/"
	mogrify images/train/*.png
	mogrify images/test/*.png

clean-downloaded:
	@echo "Cleaning images"
	rm -rf data

clean-youtube:
	@echo "Clean YouTube frames and videos..."
	rm -rf vid_data/frames vid_data/videos

clean-dataset:
	@echo "Clean images/..."
	rm -rf images
	rm *.record

clean: clean-downloaded clean-youtube clean-dataset