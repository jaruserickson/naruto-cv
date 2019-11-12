""" Annotation functions """
import json
import shutil
from os import listdir, remove
from xml.dom import minidom
from collections import defaultdict

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
    # Extract info from included annotations folder.
    extract_annotation_info()

