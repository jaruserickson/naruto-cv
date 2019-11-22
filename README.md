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
 - At this point there's a lot of stuff to set up and do to get the dataset running, so I'm going to create either a Makefile or a shell file that runs everything in order. This needs to be done for the face detector as well, since its setup was pretty beefy.
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
 - I'm going to also write a different script to run the object detector, and rather than simply output frame by frame, process the video and output that (with sound! (hopefully))
#### Village Symbol Recognition
 - Not started.
#### Main Application
 - Goal was to create a simple gui which would display a video as we processed it, as well as help the debugging process once we get to testing using videos.
 - Tried to implement the gui using Qt, but this ended up requiring more work to install and learn their video player widgets than it did to create my own OpenCV based version
 - Also tried a MatPlotLib version, but live plotting is not well supported in MatPlotLib, and the solution was too slow to be feasible.
 - Currently there is a working OpenCV based gui and backend which sends video frames to the algo as needed and provides some simple commands via keyboard shortcuts to play/pause the video, etc. The gui can also easily be extended.
