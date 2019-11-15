# naruto-cv/detect-face

So I'm going to go with the [YOLO network](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006) for a number of reasons.
 - It's incredibly fast, and as such will be useful when processing videos.
 - It uses the entire image to determine the location of the bounding box, which is important since the face alone of characters in Naruto can end up looking similar - their hair and clothes also hold weight in their identity.