""" Extract the frames for use in training
Assumes validated_frames.csv exists.
Assumes youtube_frame_validator has been run with validate=0.
    i.e. vid_data/videos is populated
"""
import os
import csv
from collections import defaultdict
import cv2

def extract_validated():
    """ Get validated frames and extract them to vid_Data/frames """
    # Get validated frames.
    frames = defaultdict(list)
    with open('validated_frames.csv', 'r') as infile:
        reader = csv.reader(infile)
        for i, line in enumerate(reader):
            if i > 0:
                frames[line[0]].append(int(line[1]))
    # Extract frames.
    for i, video in enumerate(os.listdir('vid_data/videos')):
        vid_path = os.path.join('vid_data/videos', video)
        vid_cap = cv2.VideoCapture(vid_path)
        success, image = vid_cap.read()
        count = 0
        while success:
            if count in frames[vid_path]:
                cv2.imwrite(f'vid_data/frames/{i}_{count}.jpg', image)
            success, image = vid_cap.read()
            count += 1
        vid_cap.release()


if __name__ == '__main__':
    if not os.path.isdir('vid_data/frames'):
        os.mkdir('vid_data/frames')

    extract_validated()