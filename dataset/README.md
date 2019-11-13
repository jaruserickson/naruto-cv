# naruto-cv/dataset

## Notes
There's a bunch of files here, so I'm going to list their utility as such:
1. Image validation
    - `get-image-links.sh` - Generate a file "google_images.txt" using `characters.txt` as search terms for `image_validator.py` to ingest. It will output `validated_images.json`
    - `image_validator.py` will open a GUI to manually validate images from google images. ![](../readme-images/image_validator.png)
2. Image downloading
    - `download_mal.py` - Downloads images from the characters listed in `characters.json`
    - `download_validated.py` - Downloads prevalidated images from the step above.
3. Annotations
    - `/annotations` - manually created from [labelImg](https://github.com/tzutalin/labelImg) given the above images.
    - `get_annotations_file.py` - ingests `/annotations` and outputs `annotations.json`

Therefore, the included pre-generated output files are:
 - `validated_images.json`
 - `annotations.json`

## Usage
To download the full dataset for usage locally, run the following:
```
python3 download_mal.py
python3 download_validated.py
```
