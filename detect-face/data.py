""" Dataset formatting for our CNN. """
import json
import os
import numpy as np
import cv2

IMG_INPUT_SIZE = 448

def create_dataset(test_split=0.2):
    """ Create the dataset for use in tensorflow.
        Assumes the dataset has been created via dataset/main.py.

        Keyword arguments:
        test_split  -- (float) Split the data for training and testing.
    """
    img_set, class_set = [], []
    dataset_dir = '../dataset'
    # Load in annotations
    with open(f'{dataset_dir}/annotations.json', 'r') as ann_file:
        annotations = json.load(ann_file)

    # Read in images
    data_dir = f'{dataset_dir}/data'
    for char in os.listdir(data_dir):
        for imgfile in os.listdir(f'{data_dir}/{char}'):
            img = cv2.imread(f'{data_dir}/{char}/{imgfile}')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Use RGB image.
            bndboxes = annotations[char][imgfile.split('.')[0]]
            for box in bndboxes:  # There can be more than one bounding box.
                # Add data to sets
                x_scale = IMG_INPUT_SIZE / img.shape[1]
                y_scale = IMG_INPUT_SIZE / img.shape[0]

                resized = cv2.resize(img, (IMG_INPUT_SIZE, IMG_INPUT_SIZE))
                resizebox = {
                    'name': box['name'],
                    'xmax': int(np.round(int(box['xmax']) * x_scale)),
                    'ymax': int(np.round(int(box['ymax']) * y_scale)),
                    'xmin': int(np.round(int(box['xmin']) * x_scale)),
                    'ymin': int(np.round(int(box['ymin']) * y_scale))}
                img_set.append(resized)
                class_set.append(resizebox)

    # Split data
    cut = round(len(img_set) * test_split)
    test_data = (np.array(img_set[:cut]), np.array(class_set[:cut]))
    train_data = (np.array(img_set[cut:]), np.array(class_set[cut:]))

    # Each return is (img_set, class_set)
    return train_data, test_data


# Display usage.
if __name__ == '__main__':
    # Get dataset. (cached since this isn't a real case)
    TRAIN_DS, _ = create_dataset()

    while True:
        # Plot the randomly chosen image.
        import matplotlib.pyplot as plt
        choice = np.random.randint(len(TRAIN_DS[0]))
        im = TRAIN_DS[0][choice]
        box = TRAIN_DS[1][choice]
        print(box)
        plt.imshow(cv2.rectangle(im, (box['xmax'], box['ymax']), (box['xmin'], box['ymin']), (255, 0, 0), 2))
        plt.title(box['name'])
        plt.axis('off')
        plt.show()
