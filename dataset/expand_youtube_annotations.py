""" Script to export new bounding boxes for new size video. """
import os
import xml.etree.ElementTree as ET

if not os.path.isdir('vid_data/annotations_xp'):
    os.mkdir('vid_data/annotations_xp')

# Change these for desired results.
START_WIDTH = 640
START_HEIGHT = 360
END_WIDTH = 1280
END_HEIGHT = 720

W_RATIO = END_WIDTH / START_WIDTH
H_RATIO = END_HEIGHT / START_HEIGHT

for ann in os.listdir('vid_data/annotations'):
    tree = ET.parse(os.path.join('vid_data/annotations/', ann))
    root = tree.getroot()
    # Multiply bounds by ratio
    for sz in root.iter('size'):
        for elem in sz.iter('width'):
            elem.text = str(END_WIDTH)
        for elem in sz.iter('height'):
            elem.text = str(END_HEIGHT)
    for obj in root.iter('object'):
        for bbox in obj.iter('bndbox'):
            for elem in bbox.iter('xmin'):
                elem.text = str(round(int(elem.text) * W_RATIO))
            for elem in bbox.iter('ymin'):
                elem.text = str(round(int(elem.text) * H_RATIO))
            for elem in bbox.iter('xmax'):
                elem.text = str(round(int(elem.text) * W_RATIO))
            for elem in bbox.iter('ymax'):
                elem.text = str(round(int(elem.text) * H_RATIO))
    with open(f'vid_data/annotations_xp/{ann}', 'wb') as outfile:
        s = ET.tostring(root)
        outfile.write(s)
