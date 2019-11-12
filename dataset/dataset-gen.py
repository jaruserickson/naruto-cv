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


def propogate_annotations():
    """ Propogate annotations files from annotations folder into the data folder """
    for char in listdir('./data'):
        for fname in listdir(f'./annotations/{char}'):
            shutil.copy2(f'./annotations/{char}/{fname}', f'./data/{char}')

if __name__ == '__main__':
    if not path.isdir('./data'):
        mkdir('./data')
    # Download character images.
    get_MAL_characters()
    propogate_annotations()
