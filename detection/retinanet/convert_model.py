import argparse
import os
import numpy as np
from tensorflow import keras

from model import BBoxPrediction
from keras_retinanet.util import UpsampleLike, RegressBoxes, FilterDetections, PriorProbability, ClipBoxes
from keras_retinanet.losses import smooth_l1, focal
from keras_retinanet.anchors import Anchors, AnchorParameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K = keras.backend


def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for converting a training model to an inference model.')

    parser.add_argument('model_in', help='The model to convert.')
    parser.add_argument('model_out', help='Path to save the converted model to.')
    parser.add_argument('--backbone', help='The backbone of the model to convert.', default='resnet50')
    parser.add_argument('--no-nms', help='Disables non maximum suppression.', dest='nms', action='store_false')
    parser.add_argument('--no-class-specific-filter', help='Disables class specific filtering.', dest='class_specific_filter', action='store_false')
    parser.add_argument('--config', help='Path to a configuration parameters .ini file.')

    return parser.parse_args(args)

def convert_model(model):
    """ Convert training model to prediction model. """
    anchor_params = AnchorParameters(
        sizes=[32, 64, 128, 256, 512],
        strides=[8, 16, 32, 64, 128],
        ratios=np.array([0.5, 1, 2], K.floatx()),
        scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()))
    return BBoxPrediction(model=model, anchor_params=anchor_params)

def main(args=None):
    args = parse_args(args)
    model = keras.models.load_model(args.model_in, custom_objects={
        'UpsampleLike'     : UpsampleLike,
        'PriorProbability' : PriorProbability,
        'RegressBoxes'     : RegressBoxes,
        'FilterDetections' : FilterDetections,
        'Anchors'          : Anchors,
        'ClipBoxes'        : ClipBoxes,
        '_smooth_l1'       : smooth_l1(),
        '_focal'           : focal(),
    })
    assert(all(output in model.output_names for output in ['regression', 'classification'])), \
        "Input is not a training model (no 'regression' and 'classification' outputs were found, outputs are: {}).".format(model.output_names)
    model = convert_model(model)
    model.save(args.model_out)

if __name__ == '__main__':
    main()