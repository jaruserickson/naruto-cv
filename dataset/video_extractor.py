""" Download specified video from YouTube and get X frames from it. """
import os
import shutil
import cv2
from pytube import YouTube

def get_frames(vid, framename):
    yt_path = YouTube(vid).streams.first().download()
    vid_path = shutil.move(yt_path, 'vid_data/videos')
    vid_cap = cv2.VideoCapture(vid_path)
    while vid_cap.isOpened():
        frame_num = vid_cap.get(1)
        ret, frame = vid_cap.read()
        if not ret: break
        # Every second extract a frame
        if frame_num % round(vid_cap.get(5)) == 0:
            cv2.imwrite(f'vid_data/frames/{framename}_{frame_num}.jpg', frame)
    vid_cap.release()

if __name__ == '__main__':
    if not os.path.isdir('vid_data'):
        os.mkdir('vid_data')
        os.mkdir('vid_data/videos')
        os.mkdir('vid_data/frames')

    naruto_ops = 'https://www.youtube.com/watch?v=uMyuSHewmks'
    naruto_shippuden_ops = 'https://www.youtube.com/watch?v=SHTXpNfK2R8'

    get_frames(naruto_ops, framename='naruto')
