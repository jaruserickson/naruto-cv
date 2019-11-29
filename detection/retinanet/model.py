""" RetinaNet model for object detection """
import os
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, Add, Activation, Concatenate
from keras_retinanet.util import UpsampleLike, RegressionModule, \
    ClassificationModule, FilterDetections, RegressBoxes, ClipBoxes
from keras_retinanet.anchors import Anchors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
K = keras.backend

def freeze_bn(model):
    for layer in model.layers:
        if layer.name.split('_')[-1] == 'bn':
            layer.trainable = False
    return model

def RetinaNet(anchor_params, num_classes=27, feat_size=256, modifier=None):
    """ RetinaNet based on github/keras-retinanet and the RetinaNet paper. """
    # 3 Channel image
    inputs = Input((None, None, 3))
    # Get default anchor numbers
    num_anchors = anchor_params.num_anchors()
    # We need to define our regression and classification modules:
    modules = (
        ('regression', RegressionModule(4, num_anchors, feat_size)),
        ('classification', ClassificationModule(num_classes, num_anchors, feat_size)))

    # ResNet50 as the network's backbone
    resnet = keras.applications.resnet50.ResNet50(input_tensor=inputs, weights='imagenet', include_top=False)
    resnet = freeze_bn(resnet)

    if modifier is not None:
        resnet = modifier(resnet)
    
    # We need the output of each residual conv block.
    backbone_layers = [res.output for res in resnet.layers if res.name.split('_')[-2:] == ['block3', 'out']]
    C3, C4, C5 = backbone_layers[1:]

    # Begin an FPN based on https://arxiv.org/pdf/1612.03144.pdf
    P5 = Conv2D(feat_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P5_upsampled = UpsampleLike(name='P5_upsampled')([P5, C4])
    P5 = Conv2D(feat_size, kernel_size=3, strides=1, padding='same', name='P5')(P5)

    P4 = Conv2D(feat_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P4 = Add(name='P4_merged')([P5_upsampled, P4])
    P4_upsampled = UpsampleLike(name='P4_upsampled')([P4, C3])
    P4 = Conv2D(feat_size, kernel_size=3, strides=1, padding='same', name='P4')(P4)

    P3 = Conv2D(feat_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P3 = Add(name='P3_merged')([P4_upsampled, P3])
    P3 = Conv2D(feat_size, kernel_size=3, strides=1, padding='same', name='P3')(P3)

    P6 = Conv2D(feat_size, kernel_size=3, strides=2, padding='same', name='P6')(C5)

    P7 = Activation('relu', name='C6_relu')(P6)
    P7 = Conv2D(feat_size, kernel_size=3, strides=2, padding='same', name='P7')(P7)

    # Build the pyramids based on the features above.
    build_pyramid = lambda n, m, P: Concatenate(axis=1, name=n)([m(f) for f in P])
    pyramids = [build_pyramid(n, m, [P3, P4, P5, P6, P7]) for n, m in modules]
    
    return keras.models.Model(inputs=resnet.inputs, outputs=pyramids, name='retinanet_model')

def BBoxPrediction(model, anchor_params):
    """ Reformat the trained model to output bounding boxes. A Prediction model. """
    # Extend to output the detections.
    feats = [model.get_layer(p).output for p in ['P3', 'P4', 'P5', 'P6', 'P7']]
    anchors = []
    for i, feat in enumerate(feats):
        anchors.append(Anchors(
            size=anchor_params.sizes[i],
            stride=anchor_params.strides[i],
            ratios=anchor_params.ratios,
            scales=anchor_params.scales,
            name=f'anchors_{i}')(feat))
    anchors = Concatenate(axis=1, name='anchors')(anchors)

    # Outputs from the network (bbox and classification)
    regression = model.outputs[0]
    classification = model.outputs[1]

    boxes = RegressBoxes(name='boxes')([anchors, regression])
    boxes = ClipBoxes(name='clipped_boxes')([model.inputs[0], boxes])

    detections = FilterDetections(
        nms=True,
        class_specific_filter=True,
        name='filtered_detections',
        nms_threshold=0.7,
        score_threshold=0.05
    )([boxes, classification])

    return keras.models.Model(inputs=model.inputs, outputs=detections, name='retinanet-bbox')
