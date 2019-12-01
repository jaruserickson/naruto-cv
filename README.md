# Naruto Character Recognition and Analysis

Given a scene from an episode of *Naruto*, track major character's faces, and identify the symbol on their headband.

<img src="http://i.imgur.com/70KWWkZ.png" alt="drawing" width="300"/>

### Source Clips
Some example clips we're planning to be able to run on:
 - [Naruto Shippuden Opening 18](https://www.youtube.com/watch?v=HdgD7E6JEE4) (Character variety)
 - [Naruto Opening 2](https://www.youtube.com/watch?v=SRn99oN1p_c) (Character variety)
 - [Naruto and Hinata Wedding](https://www.youtube.com/watch?v=BoMBsDIGkKI) (Character variety)
 - [Naruto vs Sasuke](https://www.youtube.com/watch?v=u_1onhckHuw) (Validation of the two main characters, and their augmentations during a fight scene.)


### Facial Detection Network
Some options:
 - [R-CNN](https://arxiv.org/pdf/1311.2524.pdf)
 - [RetinaNet](https://arxiv.org/pdf/1708.02002.pdf)
 - [YOLO](https://arxiv.org/pdf/1506.02640v5.pdf)
 - [SSD](https://arxiv.org/pdf/1512.02325.pdf)

### Usage
#### Dataset
 - While in the `dataset/` folder, to retrieve the dataset images and setup the dataset for usage in training, run `make`
 - To clean up the dataset, run `make clean`
#### Object Detection
 - Modify the paths in `dataset/training/*.config` to use your current path to `detection/` where `/home/jacob/Desktop/code/csc420/naruto-cv/detection` exists.
 - While in the `detection/` folder, after creating the dataset, to setup the models using tensorflow, run `sudo make`
   - To train the model with the data created above, run `sudo make train-frcnn` to train a Faster R-CNN model, and `sudo make train-ssd` to train an SSD model.
     - *Note*: Before training a different model, make sure to run `sudo make clean-data migrate`. This will clean up old checkpoints from the other model, so be sure to move your checkpoints if you'd like to keep them.
   - To bring up tensorboard while you train, open another teminal and run `sudo make tensorboard`
   - When training is complete, run `./export-graph.sh 1234` where `1234` is the latest checkpoint number in `HOME/TF_FOLDER/models/research/object_detection/training`
 - If you have the required modules installed (under the Makefile's install step), you can opt to run `make tf-migrate setup`
 - To clean up only the migrated datafiles from the dataset, run `sudo make clean-data`
 - To entirely clean up the detection setup, run `sudo make clean`


### Tasks
- [x] Create a dataset of character faces with tags, pulling character images from [Jikan](https://jikan.moe/), and cropping faces using the detector [lbpcascade_animeface](https://github.com/nagadomi/lbpcascade_animeface) (Note: this detector will only be used during the generation of the dataset, and only to crop faces).
   - If more character pictures are needed, we’ll manually pick them to be added from Google Images.
- [x] Train a network with the created dataset to detect and classify character faces.
  - [x] Draw a bounding box around the character faces, with their name as the tag.
- [ ] Locate and identify village symbols within each scene.
  - [ ] Determine the orientation of the found symbols and draw the corresponding box around them.
   - (Optional) If the headband symbol lays within a character bounding box, we can determine which village they are from.
   - (Optional) Detect if the village symbol on a headband has been crossed out (i.e. character is an ex-member of the village).
   - (Optional) Track characters with their respective headbands. 

### Notable Challenges
 - Some characters look really similar (e.g. Minato Namikaze, Naruto Uzumaki, Boruto Uzumaki)
 - Some characters wear their headbands in odd ways (e.g. Sakura Haruno)
 - Some characters largely change in appearance throughout the show
 - Some characters have un-unique faces (e.g. Yamato, Shino Aburame), this can increase the chance of mis-identification.
 - Modern CNNs for multiple object detection are really complex
 - Data input/output is a challenge given the custom dataset
 - Anime itself presents an interesting challenge with fight scenes with lots of movements. Both tracking and detection in "middle frames" is hard since the characters' faces are often distorted for effect.

### Notes
#### Dataset
 - I found that lpbcascade_animeface didn't work too well on Naruto characters, so I'd have to annotate them manually.
 - I decided on using [labelImg](https://github.com/tzutalin/labelImg) to annotate the images I'd retrieved.
 - While creating the dataset, when I found characters with rather "normal" faces I would include their hair as a way to add more keypoints to their classification.
 - There was around 290 pictures after using only MyAnimeList.
 - As the images from MyAnimeList probably aren't enough, I'm going to use [google_image_download](https://github.com/hardikvasa/google-images-download) to download using keywords, and validate the links manually.
 - The image validator is necessary since google images has some fail cases: ![](docs/image_validator_fail.png)
 - I found that there was some duplication and a large amount of unusable images past around 30 images on google images - so I set the limit as such. If I really need more I can try another resource.
 - There was a couple URLs the validator could see but requests couldn't download, and as such, these were blacklisted in the download shell script.
 - There's currently 765 different pictures of characters in the dataset after this.
 - After successfully running the below Faster R-CNN network on the dataset created, I've come to the conclusion that we're going to need more pictures.
 - I'll be using [pytube](https://github.com/nficano/pytube) to download a playlist I've put together (excluding the sample videos above, since those are our test set and we ideally don't want to overlap - although some overlap is inevitable) of clips of each character, then use a slightly modified validator to pick out frames I like and output them to a csv file `validated_frames.csv`. This file will be able to be fed into the youtube video downloader to extract only the frames we've validated.
   - When putting the playlist together, fan-made "AMVs" (Anime Music Videos) came in handy since there was a large amount of color correction, warps and other augmentations already embedded into the videos. They also allowed easier splitting of training and eval data.
   - The Validator ended up looking as follows. Choose frame will add the frame and filename to a CSV file `validated_frames.csv`. ![](/docs/video_frame_picker.png)
   - I went with Tkinter here since Matplotlib is actually super slow at rendering images. If I have time I'll "modernize" the google-image validator to use Tkinter UI and CSV output.
 - At this point there's a lot of stuff to set up and do to get the dataset running, so I'm going to create some Makefiles to get everything going easier.
 - The dataset now has another 488 images, to total up to **1253** images so far. It should be fairly easy to expand on this number especially with the improved framework for YouTube videos. Some characters may need more images in particular.
 - Some of the bounding boxes I made are too small for tensorflow/models, so I'm going to filter out those with an area less than `16 * 16`.
   - This didn't work, so I'm going to download larger videos instead (720p instead of 360p). Need to write a quick script to convert annotations.
 - Note*: Downloading the dataset can result in corrupt data on occasion, I need to look into a better way of keeping track of the validated google images
#### CNN Face Detection
 - Initially planned on using a YOLO CNN to detect and track characters.
 - I've done quite a bit of reading into the details of YOLO. The papers and articles of which are listed below.
   - [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)
   - [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)
   - [YOLOv3: An Incremental Improvement](https://arxiv.org/pdf/1804.02767.pdf)
   - [Understanding YOLO](https://hackernoon.com/understanding-yolo-f5a74bbc7967)
   - [YOLO - You only look once, real time object detection explained](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006)
 - YOLO seems like a good solution.
 - I'm going to take a look at some other networks.
   - [R-CNN, Fast R-CNN, Faster R-CNN, YOLO - Object Detection Algorithms](https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e)
   - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/pdf/1506.01497.pdf)
   - [Zero to Hero: Guide to Object Detection using Deep Learning: Faster R-CNN, YOLO, SSD](https://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/)
   - [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325.pdf)
 - After reading the above resources, SSD and YOLO seem like the obvious choices. 
 - I need to figure out how to input the data into these networks however, since we're using a hand-built dataset.
   -  I'd assume our best bet will be to mimick the PASCAL VOC dataset's format and use it as such.
 - I'm toying with the idea of simply following a tutorial or using a framework since the networks are quite complex (compared to U-Net, for example) when it comes down to not only the data input but their exact implementation (their layers can get quite complex) and decoding their output. I'm going to need a little more help on it.
There's a number of implementations of both YOLO and SSD on github:
 - YOLO
   - [eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3) (3k :star:)
   - [ayooshkathuria/pytorch-yolo-v3](https://github.com/ayooshkathuria/pytorch-yolo-v3) (2.3k :star:)
   - [allanzelener/YAD2K](https://github.com/allanzelener/YAD2K) (2.1k :star:)
   - [ayooshkathuria/YOLO_v3_tutorial_from_scratch](https://github.com/ayooshkathuria/YOLO_v3_tutorial_from_scratch) (1.5k :star:)
 - SSD
   - [NVIDIA Implementation of SSD](https://pytorch.org/hub/nvidia_deeplearningexamples_ssd/)
   - [balancap/SSD-Tensorflow](https://github.com/balancap/SSD-Tensorflow) (3.5k :star:)
   - [amdegroot/ssd.pytorch](https://github.com/amdegroot/ssd.pytorch) (3k :star:)
   - [sgrvinod/a-PyTorch-Tutorial-to-Object-Detection](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Object-Detection) (700 :star:)
 - Others
   - [PyTorch Implementation of Faster R-CNN](https://pytorch.org/docs/stable/torchvision/models.html#object-detection-instance-segmentation-and-person-keypoint-detection)
 - After looking into the different variants of Object detection neural networks, I've settled with the fact that it's probably better if I don't do the implementation myself. These networks are incredibly complex, with some interesting training routines, convolutional layers, input encoding, and output decoding - all of which could be points of error, let alone the architecture of the network itself. 
 - After quite a bit of looking, I ended up using the following resources to move forward with the development of our face detector:
   - [How to Train an Object Detection Classifier Using TensorFlow (GPU) on Windows 10](https://www.youtube.com/watch?v=Rgpfk6eYxJA)
   - [datitran/raccoon_dataset](https://github.com/datitran/raccoon_dataset)
   - [tensorflow/models](https://github.com/tensorflow/models)
   - [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
 - I chose to implement it this way since frameworks like Detectron don't allow for as much customization (i.e. trying out different networks easily).
 - I wrote a quick script to format the dataset for use with `tensorflow/models` (folder re-organization, CSV generation), and used the `generate_tfrecord` script from `datitran/raccoon_dataset` to generate tfrecords.
 - Some commands from the first youtube link were really helpful in getting tensorflow/models set up.
 - I had to revert tensorflow to v1.15 for some of this stuff to cooperate fully - tensorflow v2.0 is pretty new and has more limited help online than v1.x.
 - I'm going to try running a few different networks (starting with `faster_rcnn_inception_v2`) for as long as I can - it looks like we might need more than ~30 images per character however. I'm thinking we could try and aim for 100, but that's quite the time commitment for the time we have.
 - After 3 hours of training (around 12k epochs) I ran the model on the first video provided under "Source Clips" ([Naruto Shippuden Opening 18](https://www.youtube.com/watch?v=HdgD7E6JEE4))
   - ![](/docs/Loss_800images_3hrs_12000epoch.png)
   - Some pretty decent facial recognition was present, spare a few faces that showed up too often, which I would assume means there isn't enough training data for the network to appropriately determine their appearance.
   - Some characters which showed proper recognition were `naruto_uzumaki`, `kakashi_hatake`, `shikamaru_nara`, `killer_bee`, `gaara`, and `neji_hyuuga`.
     - Each of these characters had some false positives, for example, naruto would get recognized as `minato_namikaze` on occasion, and kakashi would be recognized as the body of some ninjas.
   - Characters which showed up in some ambiguous cases were: `might_guy`:: generic face, `yamato`:: generic face, `shino_aburame`:: hidden face, `konan`:: generic girl face.
 - I believe the next step should be expanding the dataset, then trying other networks.
 - I'm going to also write a different script to run the object detector, and rather than simply output frame by frame, process the video and output that (with sound!)
   - To intake some of these (the bounding boxes can get pretty small/large), i needed to change one of the parameters of `to_absolute_coordinates` `to check_range=False`.
   - To properly get sound processing working, i'm using [FFmpeg](https://github.com/FFmpeg/FFmpeg) to detatch the audio from the YouTube video and attach it to the processed video.

**Improvements**
 - Now this works - but it's not perfect. I'd like to implement OpenCV's tracking alongside the detection, and that's where a new problem arises
   - I'm going to run the detector every X frames, experimenting with what works best. I'll update the OpenCV tracker with new boxes when the detector is run. So the process for generating a video will be:
    1. Detect every X frames, replace tracking boxes
    2. Track for the next frames until detector is run
 - After adding tracking we get much better results!! A large number of the faces are correct, with about 70% label acurracy.
   - Detecting every 5 frames yields some decent results, although the detector will remove the boxes incorrectly on occasion.
   - For production purposes, I'm going to detect every 2 frames. This doubles the FPS while still retaining some detection strength

 - The other options now are a) try another more modern neural net, and b) try modifying the data (which I'm going to go with first)
   - There's a couple characters previously noted that have large fail rates, that being: `shino_aburame`, `pain`, and `neji_hyuuga`. I'm going to retain `neji_hyuuga` since I think there's enough data of him such that I can improve his detection, but I'm going to remove the other two.
 - The results are pretty good at this point: ![](/docs/detector.png)
 - Two things can still be done to improve results - trimming the character count and non-maximum supression. (And potentially data augmentation, since normal characters in light situations get recognized as other more pale characters)
 - Might have to clean out sasuke's curse mark recognition.

After omitting the problem characters, everything is looking quite a bit better.

(We never said in the proposal that we would be implementing the network, just training one LOL. I'm going to try anyway, but as mentioned before, there's a LOT of points of failure and it can get super complex.)
Now that we know the dataset is valid, we can work on writing a network from the ground up. Based on the notes before, I'm probably going to implement an SSD network.

 - [Understanding SSD MultiBox — Real-Time Object Detection In Deep Learning](https://towardsdatascience.com/understanding-ssd-multibox-real-time-object-detection-in-deep-learning-495ef744fab)
 - Region based CNNs
   - R-CNN, Fast-R-CNN, Faster-R-CNN
   - Uses an RPN (Region proposal network) to determine potential regions, then performs classification.
    - This is highly accurate but:
      - Training takes quite long (since there's multiple phases (RPN and Classifier)).
      - Network is slow when detecting after it's trained.
        - It took roughly 15 minutes to run detection on half the frames of a 2 minute 24fps video.
   - Prefferably, we'd use one network: YOLO / SSD introduce this
     - Unfortunately, using one network means we'll be looking at a substantially large number of "failure" proposals since the proposals are consistent, rather than trained (as with an RPN.)
     - SSD has higher accuracy but is a bit slower than YOLO.
       - I find that accuracy will be more important in my case, since there's quite a bit of "expression" characters can make
   - SSD (Single Shot MultiBox Detector)
     - based on the VGG-16 architecture, which is normally used for classification.
     - where fully connected layers would exist, SSD adds *auxiliary* convolutional layers, to enable extraction of features of multiple scales.
     - uses a modified "MultiBox" which is a bounding box regressino technique based on [Inception net](https://arxiv.org/abs/1409.4842).
       - has a loss function `multibox loss = confidence loss + alpha * location loss` (objectness + location)
       - starts with priors (aka "anchors") (and 1420 of them per image) which are bounding boxes that are pre-computed (so they have an IOU of over 0.5) so as to regress closer than with random boxes.
         - It retains only the top predictions with minimized location and confidence loss.
     - SSD modifies MultiBox as such:
       - for a feature map, there are manually chosen priors, with so many (the paper chose to have 6, but more will increase accuracy (while decreasing speed)) for each cell..
         - ![](https://miro.medium.com/max/1266/1*JuhjYUWXgfxMMoa4SIKLkA.png)
     - SSD uses a smooth L1-Norm for Location loss.
     - employs "Hard Negative Mining": a strategy where it keeps negative predictions at a ratio of 3:1 to the positive predictions so the network knows what isn't to be detected.
     - Random horizontal flipping and cropping (while including the target) is performed to augment the dataset.
     - In the end, Non maximum supression is performed to prune the boxes with a confidence less than some value and an IOU less than anotother (say 0.01 and 0.5 respectively) so the most likely predictions are kept. 
     - Apparently SSD struggles with objects that have a similar category - such as animals.
   - That said, SSD may not actually be the best choice. 
     - I'm going to look at something I've not yet looked at: [RetinaNet](https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4) | [Paper](https://arxiv.org/pdf/1708.02002.pdf)
       - RetinaNet proposes around 100k boxes as opposed to SSD's roughly 15k and YOLO's roughly 1k.
       - RetinaNet, while maintaining a relatively simple architecture compared to its comparatives, uses a novel loss function: Focal Loss. 
       - A ResNet is used for deep feature extraaction, with an FPN on top of ResNet for varying scales.
     - So at it's base, RetinaNet has two improvements to SSD and YOLO: the FPN and Focal Loss. 

  **Implementation**
   - I'm going to be basing my implementation largely off the well documented [keras-retinanet](https://github.com/fizyr/keras-retinanet)
   - It's got a lot of extra modularity that ends up helping a lot when training models, but we won't need any of that modularity. The documentation really helps when dissecting the repo.
     - I'm going to try and reformat the repo to a simpler format:
       - `model.py`
       - `train.py`
       - `predict.py`
     - with this, the major scope of the network should be easier to realize. (ResNet -> FPN -> Regression + Classification)
     - I'm going to use pretrained ImageNet weights for the ResNet backbone. Unfortunately since I'm going to be using tensorflow's ResNet50 implementation over the keras-resnet repo, the model will end up being slightly different. This means I can't use the original repo's (keras-retinanet) pretrained weights. This probably has a big impact on the next point.
   - After completing this, the first thing that sticks out is the slow training time.
     - *tensorflow/FRCNN* @ **0.3s/step** vs. *keras/RetinaNet* @ **2.5s/step**
     - This isn't horrible, but it will impact how long we have to train RetinaNet to recieve similar results.
     - Since our results from FRCNN were from 30,000 training steps, ideally i'd like to reach the same amount of training for RetinaNet. As such, I'll be training it for around ~15 hours.
  - [Speed Comparison](https://medium.com/@jonathan_hui/object-detection-speed-and-accuracy-comparison-faster-r-cnn-r-fcn-ssd-and-yolo-5425656ae359)
  - It actually seems after a couple of hours of training RetinaNet it's heavily overfitting. As regression converges, classification spikes super hard - this kind of makes sense since the characters look pretty similar. Will have to look into that. For now, the best implementation is that which exists in our usage of tensorflow's Faster R-CNN implementation. I might look into implementing that if this is still acting weird after another few hours.
    - If I do implement Faster R-CNN, I'm likely going to use pytorch, since it tends to run faster.

#### Village Symbol Recognition (Feature Based Matching)
 - Initially, I plan on using a Generalized Hough Transform with voting based on SIFT feature points and descriptors. Voting will be used to determine the position, scale and rotation of the symbols, along with detecting the symbol itself. Once the locations, sizes, and rotations of the symbols are known, we can re-gather the points that were used for each symbol during voting and calculate any homographies or bounding boxes using these points.
 - I have created an initial dataset of symbols using https://www.deviantart.com/makibo031226/art/NARUTO-village-symbols-521654313.
 
 **Approach 1 - Generalized Hough Transform (SIFT Based)**
 - I have now implemented a rough version of detection using interest points, however, I have come to realize some of the issues in the approach I have taken thus far:
   - The symbols I am trying to detect are too simple to gather enough feature point to do detection solely on these points, unless the symbols are quite large.
   - Using SIFT as a base for detection and matching feature descriptors is extremely slow and I will not likely be able to make it run at 24 fps even if I implement a "codebook of words" to speed up matching as SIFT itself does not run at 24 fps.
 - The results I was getting, however, were promising on images with large symbols and not too many features (i.e. a symbol on a simple background). A certain amount of visual proofs and geometric calculations were required to get the interest points to act as predictors for the Hough transform and the results were all sound. Following is a visual representation of how new interest points were matched to the stored interest points of a symbol and the corresponding prediction for each point: 
   - <img src="/docs/leaf_detect1.PNG" alt="drawing" width="350"/> 
 - Here each of the red circles are detected interest points and each of the corresponding blue circles are the predicted symbols, where the prediction includes location, relative scale, and rotation.
 - As you can see, there are very few interest points, even on such a large symbol, therefore when this is done on a smaller scale image, not enough interest points are detected. 
 - For these reasons, I will be moving to a more standard approach, I will first try using edges as the predictors (as in the usual general Hough transform), then if that doesn't work (e.g. there may not be enough edges when the symbols are small), I will try using an approach simiar to the implicit shape model by Bastian Leibe, Ales Leonardis, and Bernt Schiele - [Paper](https://link.springer.com/chapter/10.1007/11957959_26).
 
**Approach 2 - Generalized Hough Transform (Edges)**
 - I implemented my own generalized Hough transform algorithm from scratch, using OpenCV's `Sobel` and `Canny` to get edge information. The algorithm follows closely with the methods and terms in the [paper](http://www.cs.utexas.edu/~dana/HoughT.pdf), except for a few optimizations which I will elaborate more on in the following notes.
 - The first thing I noticed when I implemented the Hough transform was how slow it was. I needed the detections to be both scale and rotation invariant, but using the standard loop for the Hough transform (see [Wikipedia](https://en.wikipedia.org/wiki/Generalised_Hough_transform)) was way too slow and even small images (appr. 200x200) would take up to a minute plus if there were a lot of edges to consider. 
 - These are the optimizations I made:
   - Modified the "R-table" to include the entire set of indeces needed for each $\phi$ so that I wouldn't have to loop over the scale and rotation in order to calculate them. That is, the R-table now maps $\phi$ to the indeces in the accumulator that need to be incremented (as offsets from the location of the edge with gradient $\phi$). In the paper, the R-table maps $\phi$ to $r$ which is the offset to the location only, not inlcuding the scale and rotation. 
   - Limited the range of rotation to be from -45 degrees to 45 degrees as it is unlikely that we would need to detect beyond that.
 - Some of the problems I was facing:
   - The algorithm is still relatively slow, even after making the above optimizations as there are many edges in a single image and an extremely large number of possible positions that the symbol could be in given only that we know a single edge pixel. 
   - Some of the symbols are very generic, thus making them hard to distinguish from other objects in the image (e.g. the Mist Village symbol is simply 4 of the same slightly curved line).
   - The symbols change from one scene/image to another - that is, they are drawings and thus are not depicted exectly the same way all the time.
 - I have concluded that the scope of the inital problem was too large for a feature detection algorithm to be able to complete, at least in the amount of time we are given for this project. We initially said that the goal of this part of the project was to detect all symbols in the scene, and then match them to their corresponding characters if possible. However, given that the object we wish to detect vary in both of scale and rotation (and homography in general), and that they are not constant, it seems more reasonable to limit the scope of this problem to detecting symbols within a smaller region of the image where we know that the symbol exists. That is, from now on, the goal of symbol detection is to take a cropped image output from the face detector, and then, given the character and village, determine where and at what orientation the symbol on the headband of that character is, then place a bounding box around it. 
   - This reduces the problem from detecting which village a character is from, to only detecting the location, orientation, and size of the character's headband symbol (but doing this for multiple symbols, in a generic way). This is a better problem for feature detection to solve, as finding non-constant symbols, along with distinguishing them from each other, is not only unnecessary (given that we know the character and where their face is), but is also much more representative of a problem for deep learning to solve.
 - After experimenting with optimizations and methods on improveing detections for an extended amount of time, these what the results look like:
   - <img src="/docs/leaf_detect4.PNG" alt="drawing" width="350"/> 
 - There are still a lot of misses, especially when the frame is large, but with some tuning of the symbols (the original symbols used as templates are quite different from the actual symbols for most of the show), I think it is reasonable.
 - On average, it takes between 5 and 10 seconds for the algorithm to process one image (compared to minutes before). Also, if the input is larger than 200x200, it is resized before processing.
 - In summary, the Hough transform using edges for voting performs well enough and fast enough so that we can continue working with this as our symbol detection algorithm. It detects symbols at different scales, rotations, and can even detect them when partially occluded as well. Given that any one symbol varies from image to image, I believe this these are appropriate results.

#### Main Application
 - Goal was to create a simple gui which would display a video as we processed it, as well as help the debugging process once we get to testing using videos.
 - Tried to implement the gui using Qt, but this ended up requiring more work to install and learn their video player widgets than it did to create my own OpenCV based version
 - Also tried a MatPlotLib version, but live plotting is not well supported in MatPlotLib, and the solution was too slow to be feasible.
 - Currently there is a working OpenCV based gui and backend which sends video frames to the algo as needed and provides some simple commands via keyboard shortcuts to play/pause the video, etc. The gui can also easily be extended.
 - Noticed a flaw in the synchronization of the gui so I restructured it somewhat - i.e. there are now separate threads for reading and displaying the video. 
 - As is, we should be able to add any algo to the application and it will recieve frames as needed, while the frame reading and gui operate in a separate thread. The algo, reading, and displaying should now all be completely in sync.
