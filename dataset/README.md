# naruto-cv/dataset

## Notes
There's a bunch of files here, so I'm going to list their utility as such:
1. Image validation
    - `get-image-links.sh` - Generate a file `google_images.txt` (not pre-generated in repo) using `characters.txt` as search terms.
    - `image_validator.py` will open a GUI to manually validate images from google images.  It will output `validated_images.json`.
2. Image downloading
    - `download_mal.py` - Downloads images from the characters listed in `characters.json`
    - `download_validated.py` - Downloads prevalidated images from the step above.
3. Annotations
    - `annotations/` - manually created from [labelImg](https://github.com/tzutalin/labelImg) given the above images.
4. Dataset Creation
    - `generate_split.py` - Creates a new folder `images/` with subdirectories `test/` and `train/`, along with two CSV label files to denote annotations.

Therefore, the included pre-generated output files are:
 - `validated_images.json`

## Usage
To download the full dataset for usage locally, run the following:
```
python3 download_mal.py
python3 download_validated.py
```
To then create the usable dataset split, run:
```
python3 generate_split.py
```
The usable data can now be found at `images/`.

## Extras
If you would like to perform manual validation on the google images downloads:
```
./get-image-links.sh
python3 image_validator.py
```

#### Image Validator GUI
![](../readme-images/image_validator.png)

