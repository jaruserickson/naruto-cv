""" Create annotation file from manual annotations """
import json
from os import listdir
from xml.dom import minidom
from collections import defaultdict

def extract_annotation_info():
    """ Extract annotation info from Pascal VOC format xml to JSON for easy use. """
    get_xml_val = lambda x, y: x.getElementsByTagName(y)[0].firstChild.data
    annotations = defaultdict(lambda: defaultdict(list))
    for char in listdir('./annotations'):
        for fname in listdir(f'./annotations/{char}'):
            info = minidom.parse(f'./annotations/{char}/{fname}')

            char_obj = info.getElementsByTagName('object')  # multiple bndboxes
            for char_o in char_obj:
                bndbox = char_o.getElementsByTagName('bndbox')[0]

                ann = {
                    'name': get_xml_val(char_o, 'name'),
                    'xmin': get_xml_val(bndbox, 'xmin'),
                    'ymin': get_xml_val(bndbox, 'ymin'),
                    'xmax': get_xml_val(bndbox, 'xmax'),
                    'ymax': get_xml_val(bndbox, 'ymax')}
                # Under char's picture at fname, add the bounding box
                annotations[char][fname.split('.')[0]].append(ann)

    with open('./annotations.json', 'w') as outfile:
        json.dump(annotations, outfile)

if __name__ == '__main__':
    # Extract info from included annotations folder.
    extract_annotation_info()
