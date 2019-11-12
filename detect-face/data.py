""" Dataset formatting for our CNN. """
import json
import os
import numpy as np
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
IMG_INPUT_SIZE = 256

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
            crop = annotations[char][imgfile.split('.')[0]]
            img = crop_and_resize(img, crop, (IMG_INPUT_SIZE, IMG_INPUT_SIZE))

            # Add data to sets
            img_set.append(img)
            class_set.append(char)

    # Split data
    cut = round(len(img_set) * test_split)

    test_data = (np.array(img_set[:cut]), np.array(class_set[:cut]))
    train_data = (np.array(img_set[cut:]), np.array(class_set[cut:]))

    # Each return is (img_set, class_set)
    return train_data, test_data


def crop_and_resize(img, crop, size):
    """ Crop and resize a numpy image.

        Keyword arguments:
        img  -- (np.array) An image to crop.
        crop -- (dict) A dict containing the crop dimensions (xmin, ymin, xmax, ymax)
        size -- (tuple) A tuple containing x and y size of output image
    """
    cropped = img[int(crop['ymin']):int(crop['ymax']), int(crop['xmin']):int(crop['xmax'])]
    return cv2.resize(cropped, size)


# Display usage.
if __name__ == '__main__':
    # Get dataset.
    TRAIN_DS, _ = create_dataset()
    choice = np.random.randint(len(TRAIN_DS[0]))

    # Plot the randomly chosen image.
    import matplotlib.pyplot as plt
    plt.imshow(TRAIN_DS[0][choice])
    plt.title(TRAIN_DS[1][choice])
    plt.axis('off')
    plt.show()
