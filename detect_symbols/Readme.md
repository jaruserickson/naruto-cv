# naruto-cv/detect_symbols

## Notes
Symbol images used for the initial dataset are taken from the following image,
   ![](/docs/all_symbols.png)
   © https://masashikishimoto.deviantart.com/
   
Each symbol is cropped out and the image modified to only include the symbol on a blank background so as to make the feature detection process simpler. (Update: some of these are inaccurate, and have been updated from other images.)

To recreate the R-tables used in the Hough transform, you may run the following script:
 - `init_symbols.py` Run Canny on a directory of symbol images, then create R-tables for each of the symbols to use during detection. 
 - The script shows you the edges, with their orientations, that will be counted towards the detection. Hitting the "save" button creates the R-table and stores it to be saved when the script finishes. Hitting "next" or "previous" gives you the option to go to other symbols and save only the ones you wish to overwrite.
 - Symbols are identified by their filename, i.e. the leaf symbol will be loaded from the file `0.ext` where ext can be any valid image extension.
 
To run symbol detection, you may run `main.py` located in the root directory of the project. Run it in image mode where the all the images you wish to test it on contain as the first character of their filename, the index of the symbol you wish to detect in it.
 - e.g. `python main.py --vid-file <image-folder> --mode images`
  
These are the currently supported symbols and their id's:
 - Leaf - 0
 - Sand - 1
 - Cloud - 2
 - Stone - 3
 - Mist - 4


#### SIFT based branch
The following pertains to the branch containing the SIFT implementation of symbol detection. This version is incomplete.

In the SIFT based branch, the initialization script is somewhat different:
 - `init_symbols.py` Run SIFT on a folder of symbol images and bring up a GUI so that the user can select which SIFT keypoints they want to save to file for use in detection.


## Extras
#### Init Symbols GUI
![](/docs/init_symbols2.png)

#### Init Symbols GUI (SIFT branch)
![](/docs/init_symbols.png)
