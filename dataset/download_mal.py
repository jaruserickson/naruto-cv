""" Generate a dataset of naruto characters. """
import shutil
import time
from os import path, mkdir
import requests

# Top 30 characters from the latest character poll.
MAL_CHAR_IDS = {
    "naruto_uzumaki": 17,
    "sasuke_uchiha": 13,
    "kakashi_hatake": 85,
    "gaara": 1662,
    "itachi_uchiha": 14,
    "deidara": 1902,
    "minato_namikaze": 2535,
    "sasori": 1900,
    "shikamaru_nara": 2007,
    "hinata_hyuuga": 1555,
    "iruka_umino": 2011,
    "sakura_haruno": 145,
    "sai": 1901,
    "yamato": 2006,
    "neji_hyuuga": 1694,
    "jiraya": 2423,
    "temari": 2174,
    "rock_lee": 306,
    "kushina_uzumaki": 7302,
    "kisame_hoshigaki": 2672,
    "pain": 3180,
    "konan": 3179,
    "killer_bee": 18473,
    "might_guy": 307,
    "kiba_inuzuka": 3495,
    "ino_yamanaka": 2009,
    "shino_aburame": 3428
}

def get_MAL_characters():
    """ Retrieve character images from MAL. """
    for i, char in enumerate(MAL_CHAR_IDS.keys()):
        if i % 10 == 0 and i > 0:  # We don't want to get a timeout from MAL.
            print('Waiting for API refresh...')
            time.sleep(5)
        # Make a folder named after the character
        datapath = f'./data/{char}'
        if not path.isdir(datapath):  # If the data doesn't already exist...
            print(f'Getting data for {char} from Jikan...')
            mkdir(datapath)
            # Get images for the character
            resp = requests.get(f'https://api.jikan.moe/v3/character/{MAL_CHAR_IDS[char]}/pictures')

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

if __name__ == '__main__':
    if not path.isdir('./data'):
        mkdir('./data')
    # Download character images.
    get_MAL_characters()

