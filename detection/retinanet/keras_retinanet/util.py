"""
Utility which was slightly modified from https://github.com/fizyr/keras-retinanet
for readability and use with tf.keras over separate keras.
    - Updates to directly use TF api
    - Removal of modularity we dont need (i.e. channels_first)
    - Formatting
    - Removed reliance on old keras_resnet, moved to new TF resnet.
The core logic is largely unmodified.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, Reshape, Activation, Input
from .eval import evaluate
import numpy as np
K = keras.backend

class UpsampleLike(keras.layers.Layer):
    """ Keras layer for upsampling a Tensor to be the same shape as another Tensor. """
    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return tf.compat.v1.image.resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

class PriorProbability(keras.initializers.Initializer):
    """ Apply a prior probability to the weights. """
    def __init__(self, probability=0.01):
        self.probability = probability

    def get_config(self):
        return {'probability': self.probability}

    def __call__(self, shape, dtype=None):
        # set bias to -log((1 - p)/p) for foreground
        result = np.ones(shape) * -np.log((1 - self.probability) / self.probability)
        return result

def RegressionModule(num_values, num_anchors, feat_size, name='regression_module'):
    """ Regression module for RetinaNet. Predicts regression values upon each anchor (box).
    Modified from https://github.com/fizyr/keras-retinanet/blob/5524619f91699732ba24c6f52fb9e4b0b780b019/keras_retinanet/models/retinanet.py#L82
    """
    options = {
        'kernel_size'        : 3,
        'strides'            : 1,
        'padding'            : 'same',
        'kernel_initializer' : tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        'bias_initializer'   : 'zeros'}
    inputs = Input(shape=(None, None, feat_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=feat_size,
            activation='relu',
            name=f'pyramid_regression_{i}',
            **options
        )(outputs)

    outputs = Conv2D(num_anchors * num_values, name='pyramid_regression', **options)(outputs)
    outputs = Reshape((-1, num_values), name='pyramid_regression_reshape')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

def ClassificationModule(num_classes, num_anchors, feat_size, name='classification_module'):
    """ Classification Module for RetinaNet. Predicts classes for each anchor (box).
    Modified from https://github.com/fizyr/keras-retinanet/blob/5524619f91699732ba24c6f52fb9e4b0b780b019/keras_retinanet/models/retinanet.py#L24
    """
    inputs = Input(shape=(None, None, feat_size))
    outputs = inputs
    for i in range(4):
        outputs = Conv2D(
            filters=feat_size,
            kernel_size=3, strides=1, padding='same',
            activation='relu',
            name=f'pyramid_classification_{i}',
            kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
            bias_initializer='zeros')(outputs)

    outputs = Conv2D(
        filters=num_classes * num_anchors,
        kernel_size=3, strides=1, padding='same',
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01, seed=None),
        bias_initializer=PriorProbability(probability=0.01),
        name='pyramid_classification')(outputs)

    # reshape output and apply sigmoid
    outputs = Reshape((-1, num_classes), name='pyramid_classification_reshape')(outputs)
    outputs = Activation('sigmoid', name='pyramid_classification_sigmoid')(outputs)

    return keras.models.Model(inputs=inputs, outputs=outputs, name=name)

class FilterDetections(keras.layers.Layer):
    """ Keras layer for filtering detections using score threshold and NMS. """
    def __init__(
            self,
            nms=True,
            class_specific_filter=True,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=300,
            parallel_iterations=32,
            **kwargs):
        """ Filters detections using score threshold, NMS and selecting the top-k detections.
        Args
            nms                   : Flag to enable/disable NMS.
            class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
            nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
            score_threshold       : Threshold used to prefilter the boxes with.
            max_detections        : Maximum number of detections to keep.
            parallel_iterations   : Number of batch items to process in parallel.
        """
        self.nms = nms
        self.class_specific_filter = class_specific_filter
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.parallel_iterations = parallel_iterations
        super(FilterDetections, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        """ Constructs the NMS graph.
        Args
            inputs : List of [boxes, classification, other[0], other[1], ...] tensors.
        """
        boxes = inputs[0]
        classification = inputs[1]
        other = inputs[2:]

        # wrap nms with our parameters
        def _filter_detections(args):
            boxes = args[0]
            classification = args[1]
            other = args[2]

            return filter_detections(
                boxes,
                classification,
                other,
                nms=self.nms,
                class_specific_filter=self.class_specific_filter,
                score_threshold=self.score_threshold,
                max_detections=self.max_detections,
                nms_threshold=self.nms_threshold)

        # call filter_detections on each batch
        outputs = tf.map_fn(
            _filter_detections,
            elems=[boxes, classification, other],
            dtype=[K.floatx(), K.floatx(), 'int32'] + [o.dtype for o in other],
            parallel_iterations=self.parallel_iterations)

        return outputs

    def compute_output_shape(self, input_shape):
        """ Computes the output shapes given the input shapes.
        Args
            input_shape : List of input shapes [boxes, classification, other[0], other[1], ...].
        Returns
            List of tuples representing the output shapes:
            [filtered_boxes.shape, filtered_scores.shape, filtered_labels.shape, filtered_other[0].shape, filtered_other[1].shape, ...]
        """
        return [
            (input_shape[0][0], self.max_detections, 4),
            (input_shape[1][0], self.max_detections),
            (input_shape[1][0], self.max_detections)] \
        + [tuple([input_shape[i][0], self.max_detections] + list(input_shape[i][2:])) \
                for i in range(2, len(input_shape))]

    def compute_mask(self, inputs, mask=None):
        """ This is required in Keras when there is more than 1 output.
        """
        return (len(inputs) + 1) * [None]

    def get_config(self):
        """ Gets the configuration of this layer.
        Returns
            Dictionary containing the parameters of this layer.
        """
        config = super(FilterDetections, self).get_config()
        config.update({
            'nms'                   : self.nms,
            'class_specific_filter' : self.class_specific_filter,
            'nms_threshold'         : self.nms_threshold,
            'score_threshold'       : self.score_threshold,
            'max_detections'        : self.max_detections,
            'parallel_iterations'   : self.parallel_iterations })

        return config

def filter_detections(
        boxes,
        classification,
        other=[],
        class_specific_filter=True,
        nms=True,
        score_threshold=0.05,
        max_detections=300,
        nms_threshold=0.5):
    """ Filter detections using the boxes and classification values.
    Args
        boxes                 : Tensor of shape (num_boxes, 4) containing the boxes in (x1, y1, x2, y2) format.
        classification        : Tensor of shape (num_boxes, num_classes) containing the classification scores.
        other                 : List of tensors of shape (num_boxes, ...) to filter along with the boxes and classification scores.
        class_specific_filter : Whether to perform filtering per class, or take the best scoring class and filter those.
        nms                   : Flag to enable/disable non maximum suppression.
        score_threshold       : Threshold used to prefilter the boxes with.
        max_detections        : Maximum number of detections to keep.
        nms_threshold         : Threshold for the IoU value to determine when a box should be suppressed.
    Returns
        A list of [boxes, scores, labels, other[0], other[1], ...].
        boxes is shaped (max_detections, 4) and contains the (x1, y1, x2, y2) of the non-suppressed boxes.
        scores is shaped (max_detections,) and contains the scores of the predicted class.
        labels is shaped (max_detections,) and contains the predicted label.
        other[i] is shaped (max_detections, ...) and contains the filtered other[i] data.
        In case there are less than max_detections detections, the tensors are padded with -1's.
    """
    def _filter_detections(scores, labels):
        # threshold based on score
        indices = tf.where(K.greater(scores, score_threshold))

        if nms:
            filtered_boxes = tf.gather_nd(boxes, indices)
            filtered_scores = K.gather(scores, indices)[:, 0]
            nms_indices = tf.image.non_max_suppression(filtered_boxes, filtered_scores, max_output_size=max_detections, iou_threshold=nms_threshold)
            # filter indices based on NMS
            indices = K.gather(indices, nms_indices)

        # add indices to list of all indices
        labels = tf.gather_nd(labels, indices)
        indices = K.stack([indices[:, 0], labels], axis=1)

        return indices

    if class_specific_filter:
        all_indices = []
        # perform per class filtering
        for c in range(int(classification.shape[1])):
            scores = classification[:, c]
            labels = c * tf.ones((K.shape(scores)[0],), dtype='int64')
            all_indices.append(_filter_detections(scores, labels))

        # concatenate indices to single tensor
        indices = K.concatenate(all_indices, axis=0)
    else:
        scores = K.max(classification, axis=1)
        labels = K.argmax(classification, axis=1)
        indices = _filter_detections(scores, labels)

    # select top k
    scores = tf.gather_nd(classification, indices)
    labels = indices[:, 1]
    scores, top_indices = tf.nn.top_k(scores, k=K.minimum(max_detections, K.shape(scores)[0]))

    # filter input using the final set of indices
    indices = K.gather(indices[:, 0], top_indices)
    boxes = K.gather(boxes, indices)
    labels = K.gather(labels, top_indices)
    other_ = [K.gather(o, indices) for o in other]

    # zero pad the outputs
    pad_size = K.maximum(0, max_detections - K.shape(scores)[0])
    boxes = tf.pad(boxes, [[0, pad_size], [0, 0]], constant_values=-1)
    scores = tf.pad(scores, [[0, pad_size]], constant_values=-1)
    labels = tf.pad(labels, [[0, pad_size]], constant_values=-1)
    labels = K.cast(labels, 'int32')
    other_ = [tf.pad(o, [[0, pad_size]] + [[0, 0] for _ in range(1, len(o.shape))], constant_values=-1) for o in other_]

    # set shapes, since we know what they are
    boxes.set_shape([max_detections, 4])
    scores.set_shape([max_detections])
    labels.set_shape([max_detections])
    for o, s in zip(other_, [list(K.int_shape(o)) for o in other]):
        o.set_shape([max_detections] + s[1:])

    return [boxes, scores, labels] + other_

class RegressBoxes(keras.layers.Layer):
    """ Keras layer for applying regression values to boxes. """

    def __init__(self, mean=None, std=None, *args, **kwargs):
        """ Initializer for the RegressBoxes layer.
        Args
            mean: The mean value of the regression values which was used for normalization.
            std: The standard value of the regression values which was used for normalization.
        """
        if mean is None:
            mean = np.array([0, 0, 0, 0])
        if std is None:
            std = np.array([0.2, 0.2, 0.2, 0.2])

        if isinstance(mean, (list, tuple)):
            mean = np.array(mean)
        elif not isinstance(mean, np.ndarray):
            raise ValueError('Expected mean to be a np.ndarray, list or tuple. Received: {}'.format(type(mean)))

        if isinstance(std, (list, tuple)):
            std = np.array(std)
        elif not isinstance(std, np.ndarray):
            raise ValueError('Expected std to be a np.ndarray, list or tuple. Received: {}'.format(type(std)))

        self.mean = mean
        self.std = std
        super(RegressBoxes, self).__init__(*args, **kwargs)

    def call(self, inputs, **kwargs):
        anchors, regression = inputs
        return bbox_transform_inv(anchors, regression, mean=self.mean, std=self.std)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(RegressBoxes, self).get_config()
        config.update({
            'mean': self.mean.tolist(),
            'std' : self.std.tolist() })

        return config

class ClipBoxes(keras.layers.Layer):
    """ Keras layer to clip box values to lie inside a given shape.
    """
    def call(self, inputs, **kwargs):
        image, boxes = inputs
        shape = K.cast(K.shape(image), K.floatx())
        _, height, width, _ = tf.unstack(shape, axis=0)

        x1, y1, x2, y2 = tf.unstack(boxes, axis=-1)
        return K.stack([
            tf.clip_by_value(x1, 0, width  - 1),
            tf.clip_by_value(y1, 0, height - 1),
            tf.clip_by_value(x2, 0, width  - 1),
            tf.clip_by_value(y2, 0, height - 1)], axis=2)

    def compute_output_shape(self, input_shape):
        return input_shape[1]

def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).
    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.
    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).
    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    # Predictions
    return K.stack([
        boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width,
        boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height,
        boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width,
        boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height], axis=2)

class RedirectModel(keras.callbacks.Callback):
    """Callback which wraps another callback, but executed on a different model.
    ```python
    model = keras.models.load_model('model.h5')
    model_checkpoint = ModelCheckpoint(filepath='snapshot.h5')
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.fit(X_train, Y_train, callbacks=[RedirectModel(model_checkpoint, model)])
    ```
    Args
        callback : callback to wrap.
        model    : model to use when executing callbacks.
    """
    def __init__(self,
                 callback,
                 model):
        super(RedirectModel, self).__init__()

        self.callback = callback
        self.redirect_model = model

    def on_epoch_begin(self, epoch, logs=None):
        self.callback.on_epoch_begin(epoch, logs=logs)

    def on_epoch_end(self, epoch, logs=None):
        self.callback.on_epoch_end(epoch, logs=logs)

    def on_batch_begin(self, batch, logs=None):
        self.callback.on_batch_begin(batch, logs=logs)

    def on_batch_end(self, batch, logs=None):
        self.callback.on_batch_end(batch, logs=logs)

    def on_train_begin(self, logs=None):
        # overwrite the model with our custom model
        self.callback.set_model(self.redirect_model)

        self.callback.on_train_begin(logs=logs)

    def on_train_end(self, logs=None):
        self.callback.on_train_end(logs=logs)

class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """
    def __init__(
            self,
            generator,
            iou_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            save_path=None,
            tensorboard=None,
            weighted_average=False,
            verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.
        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions, _ = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path)

        # compute per class average precision
        total_instances = []
        precisions = []
        for label, (average_precision, num_annotations) in average_precisions.items():
            if self.verbose == 1:
                print('{:.0f} instances of class'.format(num_annotations),
                      self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)

        if self.tensorboard:
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)

        logs['mAP'] = self.mean_ap

        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))

def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.
    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (K.arange(0, shape[1], dtype=K.floatx()) \
        + K.constant(0.5, dtype=K.floatx())) * stride
    shift_y = (K.arange(0, shape[0], dtype=K.floatx()) \
        + K.constant(0.5, dtype=K.floatx())) * stride

    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    shift_x = K.reshape(shift_x, [-1])
    shift_y = K.reshape(shift_y, [-1])

    shifts = K.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y], axis=0)

    shifts = K.transpose(shifts)
    number_of_anchors = K.shape(anchors)[0]

    k = K.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = K.reshape(anchors, [1, number_of_anchors, 4]) \
        + K.cast(K.reshape(shifts, [k, 1, 4]), K.floatx())
    shifted_anchors = K.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors