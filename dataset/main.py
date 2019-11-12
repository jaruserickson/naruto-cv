""" Generate a dataset of naruto characters. """
import shutil
import json
import time
from os import path, mkdir, listdir
from xml.dom import minidom
from collections import defaultdict
import requests

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


def extract_annotation_info():
    """ Extract annotation info from Pascal VOC format xml to JSON for easy use. """
    get_xml_val = lambda x, y: x.getElementsByTagName(y)[0].firstChild.data
    annotations = defaultdict(defaultdict)
    for char in listdir('./annotations'):
        for fname in listdir(f'./annotations/{char}'):
            info = minidom.parse(f'./annotations/{char}/{fname}')

            char_obj = info.getElementsByTagName('object')[0]
            bndbox = char_obj.getElementsByTagName('bndbox')[0]

            ann = {
                'xmin': get_xml_val(bndbox, 'xmin'),
                'ymin': get_xml_val(bndbox, 'ymin'),
                'xmax': get_xml_val(bndbox, 'xmax'),
                'ymax': get_xml_val(bndbox, 'ymax')}
            annotations[get_xml_val(char_obj, 'name')][fname.split('.')[0]] = ann

    with open('./annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)


if __name__ == '__main__':
    if not path.isdir('./data'):
        mkdir('./data')
    # Download character images.
    get_MAL_characters()
    # Extract info from included annotations folder.
    extract_annotation_info()
