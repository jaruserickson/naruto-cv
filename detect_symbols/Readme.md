# naruto-cv/detect_symbols

## Notes
Symbol images used for the initial dataset are taken from the following image,
   ![](/docs/all_symbols.png)
   © https://masashikishimoto.deviantart.com/
   
Each symbol is cropped out and the image modified to only include the symbol on a blank background so as to make the feature detection process simpler.

To create a library of feature points to use in detecting symbols anywhere in an image, I created the following script which allows me to specify which feature points I want to use for detection for each symbol.
  - `init_symbols.py` Run SIFT on a folder of symbol images and bring up a GUI so that the user can select which SIFT keypoints they want to save to file for later use.


## Extras
#### Init Symbols GUI
![](/docs/init_symbols.png)
