""" Train RetinaNet model."""
import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras

from keras_retinanet.anchors import AnchorParameters
from keras_retinanet.transform import random_transform_generator
from keras_retinanet.image import random_visual_effect_generator, preprocess_image
from keras_retinanet.util import RedirectModel, Evaluate
import keras_retinanet.losses as loss
from keras_retinanet.data import CSVGenerator

from model import RetinaNet, BBoxPrediction
K = keras.backend

def parse_args(args):
    """ Parse the arguments. """
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('annotations', help='Path to CSV file containing annotations for training.')
    parser.add_argument('classes', help='Path to a CSV file containing class label mapping.')
    parser.add_argument('--val-annotations', help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--batch-size', help='Size of the batches.', default=1, type=int)
    parser.add_argument('--initial-epoch', help='Epoch from which to begin the train, useful if resuming from snapshot.', type=int, default=0)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=50)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    parser.add_argument('--lr', help='Learning rate.', type=float, default=1e-5)
    parser.add_argument('--snapshot-path', help='Path to store snapshots of models during training (defaults to \'./snapshots\')', default='./snapshots')
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output', default='./logs')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true')
    parser.add_argument('--image-min-side', help='Rescale the image so the smallest side is min_side.', type=int, default=800)
    parser.add_argument('--image-max-side', help='Rescale the image if the largest side is larger than max_side.', type=int, default=1333)
    parser.add_argument('--no-resize', help='Don''t rescale the image.', action='store_true')
    parser.add_argument('--freeze-backbone',  help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--weighted-average', help='Compute the mAP using the weighted average of precisions among classes.', action='store_true')
    parser.add_argument('--compute-val-loss', help='Compute validation loss during training', dest='compute_val_loss', action='store_true')

    return parser.parse_args(args)

def create_generators(args, preprocess_image):
    """ Create generators for training and validation.

    Args
        args             : parseargs object containing configuration for generators.
        preprocess_image : Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size'       : args.batch_size,
        'image_min_side'   : args.image_min_side,
        'image_max_side'   : args.image_max_side,
        'no_resize'        : args.no_resize,
        'preprocess_image' : preprocess_image}

    # create random transform generator for augmenting training data
    if args.random_transform:
        transform_generator = random_transform_generator(
            min_rotation=-0.1,
            max_rotation=0.1,
            min_translation=(-0.1, -0.1),
            max_translation=(0.1, 0.1),
            min_shear=-0.1,
            max_shear=0.1,
            min_scaling=(0.9, 0.9),
            max_scaling=(1.1, 1.1),
            flip_x_chance=0.5,
            flip_y_chance=0.5)
        visual_effect_generator = random_visual_effect_generator(
            contrast_range=(0.9, 1.1),
            brightness_range=(-.1, .1),
            hue_range=(-0.05, 0.05),
            saturation_range=(0.95, 1.05))
    else:
        transform_generator = random_transform_generator(flip_x_chance=0.5)
        visual_effect_generator = None

    train_generator = CSVGenerator(
        args.annotations,
        args.classes,
        transform_generator=transform_generator,
        visual_effect_generator=visual_effect_generator,
        **common_args)

    if args.val_annotations:
        test_generator = CSVGenerator(
            args.val_annotations,
            args.classes,
            shuffle_groups=False,
            **common_args)
    else:
        test_generator = None

    return train_generator, test_generator

def create_callbacks(model, prediction_model, test_generator, args):
    """ Minimized version of callback function from github. """
    callbacks = []

    tensorboard_callback = None
    # Set up tensorboard.
    if args.tensorboard_dir:
        if not os.path.isdir(args.tensorboard_dir):
            os.makedirs(args.tensorboard_dir)
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            batch_size=args.batch_size)
        evaluation = Evaluate(test_generator, tensorboard=tensorboard_callback, weighted_average=args.weighted_average)
        evaluation = RedirectModel(evaluation, prediction_model)
        callbacks.append(evaluation)

    # save the model
    # ensure directory created first; otherwise h5py will error after epoch.
    if not os.path.isdir(args.snapshot_path):
        os.makedirs(args.snapshot_path)
    checkpoint = keras.callbacks.ModelCheckpoint(
        os.path.join(
            args.snapshot_path,
            'resnet50_naruto_{epoch:02d}.h5'), verbose=1)
    checkpoint = RedirectModel(checkpoint, model)
    callbacks.append(checkpoint)

    callbacks.append(keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        patience=2,
        verbose=1))

    if args.tensorboard_dir:
        callbacks.append(tensorboard_callback)

    return callbacks

def freeze_model(model):
    """ Set all layers in a model to non-trainable.
    The weights for these layers will not be updated during training.
    This function modifies the given model in-place,
    but it also returns the modified model to allow easy chaining with other functions.
    """
    for layer in model.layers:
        layer.trainable = False
    return model

def setup_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

def main(args=None):
    """ Modified version of the train function from github. """
    args = parse_args(args)
    # setup_gpu()
    train_generator, test_generator = create_generators(args, preprocess_image)
    anchor_params = AnchorParameters(
        sizes=[32, 64, 128, 256, 512],
        strides=[8, 16, 32, 64, 128],
        ratios=np.array([0.5, 1, 2], K.floatx()),
        scales=np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], K.floatx()))
    
    modifier = freeze_model if args.freeze_backbone else None

    # create the model
    if args.snapshot is not None:
        print('Loading model, this may take a second...')
        from keras_retinanet.util import UpsampleLike, RegressBoxes, FilterDetections, PriorProbability, ClipBoxes
        from keras_retinanet.losses import smooth_l1, focal
        from keras_retinanet.anchors import Anchors
        model = keras.models.load_model(args.snapshot, \
            custom_objects={
                'UpsampleLike'     : UpsampleLike,
                'PriorProbability' : PriorProbability,
                'RegressBoxes'     : RegressBoxes,
                'FilterDetections' : FilterDetections,
                'Anchors'          : Anchors,
                'ClipBoxes'        : ClipBoxes,
                '_smooth_l1'       : smooth_l1(),
                '_focal'           : focal(),
            })
        training_model = model
        prediction_model = BBoxPrediction(model=model, anchor_params=anchor_params)
    else:
        print('Creating model, this may take a second...')
        model = RetinaNet(anchor_params, modifier=modifier)
        training_model = model
        prediction_model = BBoxPrediction(model=model, anchor_params=anchor_params)

        training_model.compile(
            loss={'regression': loss.smooth_l1(), 'classification': loss.focal()},
            optimizer=keras.optimizers.Adam(lr=args.lr, clipnorm=0.001))

    print(model.summary())

    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        test_generator,
        args)

    if not args.compute_val_loss:
        test_generator = None

    # start training
    return training_model.fit_generator(
        generator=train_generator,
        steps_per_epoch=args.steps,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
        use_multiprocessing=False,
        validation_data=test_generator,
        initial_epoch=args.initial_epoch)

if __name__ == '__main__':
    main()
