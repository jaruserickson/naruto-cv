""" Create usable split for TFRecord generation """
import os
import shutil
import csv
from xml.dom import minidom
from PIL import Image

def create_folders():
    """ Create folders to split data into """
    if not os.path.isdir('images'):
        # Create folders if they don't exist
        os.mkdir('images')

        if not os.path.isdir('images/train'):
            os.mkdir('images/train')
        if not os.path.isdir('images/test'):
            os.mkdir('images/test')

def propogate_data(split=0.9):
    """ Propogate data from data and annotations into the folders. """
    # Validate folder existence
    if not os.path.isdir('images'):
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
        shutil.copy2(img_path, os.path.join('images/train', '_'.join(img_path.split('/')[-2:])))
    for img_path in test_paths:
        shutil.copy2(img_path, os.path.join('images/test', '_'.join(img_path.split('/')[-2:])))

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
        shutil.copy2(img_path, os.path.join('images/train', img_path.split('/')[-1]))
    for img_path in test_paths:
        shutil.copy2(img_path, os.path.join('images/test', img_path.split('/')[-1]))

def not_included_chars():
    """ Get include=False charactersfrom the csv. """
    characters = []
    with open('characters.csv', 'r') as infile:
        reader = csv.reader(infile)
        for i, line in enumerate(reader):
            if i > 0 and line[2] == 'False':
                print(f'{line[0]} was omitted in characters.csv.')
                characters.append(line[0])
    return characters

def get_csv_labels(folder):
    """ Get CSV labels from annotation files"""
    get_xml_val = lambda x, y: x.getElementsByTagName(y)[0].firstChild.data
    header = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    labels = [header]
    omit_chars = not_included_chars()
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
                    labels.append([_filename, _width, _height, _class, _xmin, _ymin, _xmax, _ymax])
    return labels

def generate_csv_label_files():
    """ Generate CSV files for TFRecord consumption """
    train_labels = get_csv_labels('images/train')
    test_labels = get_csv_labels('images/test')

    with open('images/train_labels.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(train_labels)
    with open('images/test_labels.csv', 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(test_labels)


if __name__ == '__main__':
    print('Creating folders')
    create_folders()
    print('Propogating data')
    propogate_data()
    print('Generating CSV files')
    generate_csv_label_files()
    