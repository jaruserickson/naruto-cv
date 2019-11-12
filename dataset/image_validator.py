""" Validate images for use in the dataset.
Assumes that get-image-links.sh was run.
Keep their links noted in dataset_images.json.
"""
import json
import os
from collections import defaultdict
from skimage import io
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

def get_google_images_dict(filename='./google_images.txt'):
    """ Dump google image downloader's output into a dictionary """
    with open(filename, 'r') as googleimages:
        images = googleimages.readlines()

    cleaned = defaultdict(list)
    char = ''
    for line in images:
        line = line.strip()
        # Character line.
        if 'Item no.:' in line:
            char = line.split('Item name = ')[-1].split(' ')
            if char[-1] == 'naruto':  # Clean out the "show" keyword
                char = char[:-1]
            char = '_'.join(char)

        # Image line:
        if 'Image URL:' in line:
            line = line.split('Image URL: ')
            url = line[-1]
            cleaned[char].append(url)

    return cleaned

class Index(object):
    """ Class index for use in the validator GUI """
    def __init__(self, images, char, obj):
        self.images = images
        if os.path.isfile('validated_images.json'):
            with open('validated_images.json', 'r') as infile:
                images = json.load(infile)
                if char in images.keys():
                    self.validated = images[char]
                else:
                    self.validated = []
        else:
            self.validated = []
        self.char = char
        self.obj = obj
        self.ind = 0

    def next(self, _):
        self.ind += 1
        print(f'{self.ind}/{len(self.images)}')
        if self.ind < len(self.images):
            try:
                img = io.imread(self.images[self.ind])
                self.obj.set_data(img)
                plt.draw()
            except Exception:
                print('Failure with next image. Skipping.')
                self.next(_)
        else:
            print('Reached end of images.')

    def validate(self, _):
        if not self.images[self.ind] in self.validated:
            print('Validated image.')
            self.validated.append(self.images[self.ind])
            self.next(_)
        else:
            print('Image already validated.')

    def prev(self, _):
        self.ind -= 1
        print(f'{self.ind}/{len(self.images)}')
        if self.ind > 0:
            try:
                img = io.imread(self.images[self.ind])
                self.obj.set_data(img)
                plt.draw()
            except Exception:
                print('Failure with prev image. Skipping.')
                self.prev(_)
        else:
            print('Cannot go previous to the first image!')

    def write(self):
        if os.path.isfile('validated_images.json'):
            with open('validated_images.json', 'r') as infile:
                images = json.load(infile)
        else:
            images = defaultdict(list)
        images[self.char] = self.validated
        with open('validated_images.json', 'w') as outfile:
            json.dump(images, outfile)


def validate_images():
    """ Go through the links, and run UI validation.
        Outputs validated_images.json with validated images
    """
    downloaded = get_google_images_dict()
    for char in downloaded.keys():
        images = downloaded[char]
        img = io.imread(images[0])
        obj = plt.imshow(img)
        plt.title(char)
        plt.axis('off')
        callback = Index(images, char, obj)

        bnext = Button(plt.axes([0.82, 0.05, 0.1, 0.075]), 'Next')
        bnext.on_clicked(callback.next)
        bval = Button(plt.axes([0.71, 0.05, 0.1, 0.075]), 'Validate')
        bval.on_clicked(callback.validate)
        bprev = Button(plt.axes([0.6, 0.05, 0.1, 0.075]), 'Previous')
        bprev.on_clicked(callback.prev)
        plt.show()
        callback.write()


if __name__ == '__main__':
    validate_images()