""" Download images validated by image_validator """
import json
import os
import shutil
import requests

def download_images():
    """ Download prevalidated images, noted in dataset_images.json """
    with open('validated_images.json', 'r') as infile:
        chars = json.load(infile)

    for char in chars.keys():
        datapath = f'./data/{char}'
        startpoint = int(sorted(os.listdir(datapath))[-1].split('.')[0]) + 1
        for img in chars[char]:
            resp = requests.get(img, stream=True)
            if resp.status_code == 200:
                with open(f'{datapath}/{startpoint}.png', 'wb') as out_file:
                    print(f'Downloading to {datapath}/{startpoint}.png')
                    shutil.copyfileobj(resp.raw, out_file)
                    startpoint += 1
            else:
                print(f"Couldn't get image from {img}.")

if __name__ == '__main__':
    download_images()
