""" Download specified video from YouTube and get X frames from it. """
import os
import shutil
import csv
import sys
import tkinter as tk
from PIL import Image, ImageTk
import cv2
from pytube import YouTube
import pandas as pd

class Validator():
    """ Video Validator UI. """
    def __init__(self, window, vid_path):
        self.count = 0
        self.validated = []
        self.vid_path = vid_path
        self.vid_cap = cv2.VideoCapture(self.vid_path)
        self.success, self.image = self.vid_cap.read()
        window.title(vid_path.split('/')[-1])
        self.window = window

        img = self.convert_opencv_to_tk(self.image)
        self.label = tk.Label(self.window)
        self.label.grid(row=0, column=0)
        self.label.configure(image=img)
        self.label.image = img

        frame = tk.Frame(self.window)
        frame.grid(row=1, column=0)
        self.b_next = tk.Button(
            frame, text='Next Frame', command=self.on_next_frame)
        self.b_choose = tk.Button(
            frame, text='Choose Frame', command=self.on_choose_frame)
        self.b_next.pack(side=tk.RIGHT)
        self.b_choose.pack(side=tk.LEFT)

        self.b_vid = tk.Button(
            self.window, text='Next Video', bg='red', command=self.on_next_video)
        self.b_vid.grid(row=2, column=0)

    def convert_opencv_to_tk(self, img):
        """ Convert OpenCV image to TkImage. """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return ImageTk.PhotoImage(image=Image.fromarray(img))

    def on_next_frame(self):
        """ Scrub to next frame. """
        self.success, self.image = self.vid_cap.read()
        self.count += 1
        while self.count % round(self.vid_cap.get(cv2.CAP_PROP_FPS)) != 0:
            self.success, self.image = self.vid_cap.read()
            self.count += 1
        if not self.success:
            self.window.destroy()

        percentage = self.count / self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT) * 100
        print(f'Progress: {round(percentage, 2)}%', end='\r')

        # Set the image in the Tkinter window.
        img = self.convert_opencv_to_tk(self.image)
        self.label.configure(image=img)
        self.label.image = img

    def on_choose_frame(self):
        """ Choose the current frame for training """
        print(f'Chosen: [{self.count}]\n')
        self.validated.append([self.vid_path, self.count])
        self.on_next_frame()  # Move to next frame.

    def on_next_video(self):
        """ Destroy Tkinter window to move to next video. """
        self.window.destroy()

    def get_validated(self):
        """ Getter for validated set. """
        return self.validated

    def release(self):
        """ Release vid cap """
        self.vid_cap.release()


def validate_videos(vid_links, to_validate):
    """ Download YouTube vid_links, and validate with GUI. """
    csv_header = ['vid_location', 'frame_number']
    validated = [csv_header]

    for vid, name in vid_links:
        # Download the video only if it doesn't already exist
        if name not in os.listdir('vid_data/videos'):
            print(f'Downloading {name} from YouTube.')
            yt_path = YouTube(vid).streams.filter(
                res='360p', mime_type='video/mp4').first().download()
            vid_path = shutil.move(yt_path, 'vid_data/videos')
        else:
            print(f'{name} already downloaded.')
            vid_path = os.path.join('vid_data/videos', name)
        # Only run validation software if requested
        if to_validate:
            # Set up Tkinter window
            root = tk.Tk()
            val = Validator(root, vid_path)
            root.mainloop()

            # On close, loop around and use the next video.
            print('Closing, moving onto next video..')
            validated.extend(val.get_validated())
            val.release()

    # Output results to CSV
    if to_validate and len(validated) > 1:
        print('Saving to CSV...')
        with open('validated_frames.csv', 'a') as outfile:
            writer = csv.writer(outfile)
            if os.path.isfile('validated_frames.csv'):
                validated = validated[1:]
            writer.writerows(validated)
        # Remove duplicates.
        df = pd.read_csv('validated_frames.csv', index_col=False)
        df.drop_duplicates(inplace=True)
        df.to_csv('validated_frames.csv', index=False)
    elif len(validated) == 1:
        print('Nothing validated. Closing.')
    else:
        print('Closing.')


if __name__ == '__main__':
    VALIDATE_VIDEOS = True if len(sys.argv) == 1 else int(sys.argv[1])

    if not os.path.isdir('vid_data'):
        os.mkdir('vid_data')
        os.mkdir('vid_data/videos')

    vid_links = [
        ('https://www.youtube.com/watch?v=hWp61OEy4R4', 'NARUTO IN 18 MINUTES.mp4'),
        ('https://www.youtube.com/watch?v=IER6REy5qtw', 'NARUTO SHIPPUDEN IN 15 MINUTES「REVAMPED VERSION」.mp4')
    ]

    validate_videos(vid_links, VALIDATE_VIDEOS)
