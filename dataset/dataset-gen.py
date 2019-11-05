""" Generate a dataset of naruto characters. """
import cv2
import sys
from os import path, mkdir, listdir
import requests
import shutil
import json
import time

# Top 30 characters from the latest character poll.
with open('characters.json', 'r') as characters:
    CHARACTER_ID = json.load(characters)

def get_MAL_characters():
    """ Retrieve character images from MAL. """
    for i, char in enumerate(CHARACTER_ID.keys()):
        if i % 10 == 0 and i > 0:  # We don't want to get a timeout from MAL.
            print('Waiting for API refresh...')
            time.sleep(5)
        # Make a folder named after the character
        datapath = f'./data/{char}'
        if not path.isdir(datapath):  # If the data doesn't already exist...
            print(f'Getting data for {char} from Jikan...')
            mkdir(datapath)
            # Get images for the character
            resp = requests.get(f'https://api.jikan.moe/v3/character/{CHARACTER_ID[char]}/pictures')
            
            # Save the images of the character
            if resp.status_code == 200:
                pictures = resp.json()['pictures']
                print(f'Downloading images for {char} from MAL...')
                for i, pic in enumerate(pictures):
                    resp = requests.get(pic['large'], stream=True)
                    if resp.status_code == 200:
                        with open(f'{datapath}/{i}.png', 'wb') as out_file:
                            shutil.copyfileobj(resp.raw, out_file)
                    else:
                        raise RuntimeError(f"Couldn't get image {pic} from MAL.")
            else:
                raise RuntimeError(f"Couldn't get character data of '{char}' from Jikan.")
        else:
            print(f'Skipping {char}. Data already downloaded.')


def detect(filename, cascade_file="./lbpcascade_animeface.xml"):
    """ Detect anime faces in an image provided using OpenCV Cascade.

        Keyword arguments:
        filename    -- (str) filename to detect upon
        cascade_file -- (str) provided animeface cascade_file
    """
    if not path.isfile(cascade_file):
        raise RuntimeError(f'{cascade_file}: not found')

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))

    for x, y, w, h in faces:  # Face bounds
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('Face Detection', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Download character images.
    get_MAL_characters()
    # Go through all the images downloaded and detect.
    for char in listdir('./data'):
        for img in listdir(path.join('./data', char)):
            detect(path.join('./data', char, img))