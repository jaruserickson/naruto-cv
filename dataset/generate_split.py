""" Create usable split for TFRecord generation """
import os
import sys
import shutil
import csv
from xml.dom import minidom
from PIL import Image

def create_folders(folder='images'):
    """ Create folders to split data into """
    if not os.path.isdir(folder):
        # Create folders if they don't exist
        os.mkdir(folder)

        if not os.path.isdir(f'{folder}/train'):
            os.mkdir(f'{folder}/train')
        if not os.path.isdir(f'{folder}/test'):
            os.mkdir(f'{folder}/test')

def propogate_data(folder='images', split=0.9):
    """ Propogate data from data and annotations into the folders. """
    # Validate folder existence
    if not os.path.isdir(folder):
        raise Exception('create_folders must be run prior to this function.')
    if not os.path.isdir('data'):
        raise Exception('data folder must exist to create a split.')
    if not os.path.isdir('annotations') or not os.path.isdir('vid_data/annotations'):
        raise Exception('annotations must exist for a proper data split.')
    if not os.path.isdir('vid_data/frames'):
        raise Exception('video frames must exist for a proper data split.')

    train_paths, test_paths = [], []

    # Get imagepaths from data/
    for char_folder in os.listdir('data'):
        char_path = os.path.join('data', char_folder)
        ann_path = os.path.join('annotations', char_folder)
        split_index = int(round(len(os.listdir(char_path)) * split))

        for img in os.listdir(char_path)[:split_index]:
            train_paths.append(os.path.join(char_path, img))
            ann_file = os.path.join(ann_path, img.split('.')[0] + '.xml')
            train_paths.append(ann_file)
        for img in os.listdir(char_path)[split_index:]:
            test_paths.append(os.path.join(char_path, img))
            ann_file = os.path.join(ann_path, img.split('.')[0] + '.xml')
            test_paths.append(ann_file)

    for img_path in train_paths:
        dir_path, fname = os.path.split(img_path)
        char_name = os.path.split(dir_path)[-1]
        shutil.copy2(img_path, os.path.join(f'{folder}/train', char_name + '_' + fname))
    for img_path in test_paths:
        dir_path, fname = os.path.split(img_path)
        char_name = os.path.split(dir_path)[-1]
        shutil.copy2(img_path, os.path.join(f'{folder}/test', char_name + '_' + fname))

    # Get imagepaths from vid_data/
    filenames = [x.split('.png')[0] for x in os.listdir('vid_data/frames')]
    split_index = int(round(len(filenames) * split))
    train_paths, test_paths = [], []
    for fname in filenames[:split_index]:
        train_paths.append(os.path.join('vid_data/frames', f'{fname}.png'))
        train_paths.append(os.path.join('vid_data/annotations', f'{fname}.xml'))
    for fname in filenames[split_index:]:
        test_paths.append(os.path.join('vid_data/frames', f'{fname}.png'))
        test_paths.append(os.path.join('vid_data/annotations', f'{fname}.xml'))

    for img_path in train_paths:
        shutil.copy2(img_path, os.path.join(f'{folder}/train', os.path.split(img_path)[1]))
    for img_path in test_paths:
        shutil.copy2(img_path, os.path.join(f'{folder}/test', os.path.split(img_path)[1]))

def include_chars(include=False):
    """ Get include={include} characters from the csv. """
    characters = []
    with open('characters.csv', 'r') as infile:
        reader = csv.reader(infile)
        for i, line in enumerate(reader):
            if i > 0 and line[2] == str(include):
                characters.append(line[0])
    return characters

def get_csv_labels(folder, order):
    """ Get CSV labels from annotation files. order = FRCNN | Retina """
    get_xml_val = lambda x, y: x.getElementsByTagName(y)[0].firstChild.data
    header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    labels = [header] if order == 'FRCNN' else []
    omit_chars = include_chars(False)
    for f_name in os.listdir(folder):
        # Only create entries with XML files.
        if f_name.split('.')[-1] == 'xml':
            info = minidom.parse(os.path.join(folder, f_name))
            sizeobj = info.getElementsByTagName('size')[0]
            _filename = f"{f_name.split('.')[0]}.png"
            _width = int(get_xml_val(sizeobj, 'width'))
            _height = int(get_xml_val(sizeobj, 'height'))
            # If the width/height wasn't read by labelImg, we have to do it.
            if _width == 0 or _height == 0:
                _width, _height = Image.open(os.path.join(folder, _filename)).size
                _width, _height = int(_width), int(_height)

            char_obj = info.getElementsByTagName('object')  # multiple bndboxes
            for char_o in char_obj:
                bndbox = char_o.getElementsByTagName('bndbox')[0]
                _class = get_xml_val(char_o, 'name')
                _xmin = int(get_xml_val(bndbox, 'xmin'))
                _ymin = int(get_xml_val(bndbox, 'ymin'))
                _xmax = int(get_xml_val(bndbox, 'xmax'))
                _ymax = int(get_xml_val(bndbox, 'ymax'))
                # Filter out bounding boxes which are too small.
                # Small bounding boxes cause issues with Tensorflow.
                w = (_xmax - _xmin)
                h = (_ymax - _ymin)
                # Perform the filter.
                if w < 33 or h < 33:
                    print(f'BBox of {_filename} too small!')
                elif _class in omit_chars:
                    continue
                else:
                    # Add bndbox to row
                    if order == 'FRCNN':
                        labels.append([_filename, _width, _height, _class, _xmin, _ymin, _xmax, _ymax])
                    elif order == 'Retina':
                        labels.append([os.path.abspath(os.path.join('../dataset', folder, _filename)), _xmin, _ymin, _xmax, _ymax, _class])
                    else:
                        assert ValueError('order must be one of Retina or FRCNN')
    return labels

def generate_csv_label_files(folder='images', order='FRCNN'):
    """ Generate CSV files for TFRecord consumption """
    train_labels = get_csv_labels(f'{folder}/train', order)
    test_labels = get_csv_labels(f'{folder}/test', order)

    with open(f'{folder}/train_labels.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(train_labels)
    with open(f'{folder}/test_labels.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(test_labels)

def retina_characters(folder='images'):
    """ Get characters for use in RetinaNet """
    characters = []
    with open('characters.csv', 'r') as infile:
        reader = csv.reader(infile)
        for i, line in enumerate(reader):
            characters.append([line[0], i-1])
    characters = characters[1:]
    with open(f'{folder}/characters.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(characters)

if __name__ == '__main__':
    FOLDER = 'images' if len(sys.argv) < 2 else sys.argv[1] # retimages
    ORDER = 'FRCNN' if len(sys.argv) < 3 else sys.argv[2]   # Retina
    print('Creating folders')
    create_folders(FOLDER)
    print('Propogating data')
    propogate_data(FOLDER)
    print('Generating CSV files')
    generate_csv_label_files(FOLDER, order=ORDER)
    if ORDER == 'Retina':
        retina_characters(FOLDER)
