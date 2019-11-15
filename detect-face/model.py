""" YOLOv3 Model for Detection """
from tensorflow import keras

def yolo_model(optim, loss):
    """ YOLOv3, as per the paper cited.

        Keyword arguments:
        optim   -- (keras.optimizer) Optimizer to compile upon.
        loss    -- (keras.losses) Loss function to compile upon.
    """
    inputs = keras.Input((448, 448, 3))
    layer1 = keras.Sequential([
        keras.layers.Conv2D(64, 7, activation='relu', strides=2),
        keras.layers.MaxPooling2D(2, strides=2)
    ])(inputs)

    layer2 = keras.Sequential([
        keras.layers.Conv2D(192, 3, activation='relu'),
        keras.layers.MaxPooling2D(2, strides=2)
    ])(layer1)

    layer3 = keras.Sequential([
        keras.layers.Conv2D(128, 1, activation='relu'),
        keras.layers.Conv2D(256, 3, activation='relu'),
        keras.layers.Conv2D(256, 1, activation='relu'),
        keras.layers.Conv2D(512, 3, activation='relu'),
        keras.layers.MaxPooling2D(2, strides=2)
    ])(layer2)

    layer4 = keras.Sequential([
        keras.layers.Conv2D(256, 1, activation='relu'),
        keras.layers.Conv2D(512, 3, activation='relu'),
        keras.layers.Conv2D(256, 1, activation='relu'),
        keras.layers.Conv2D(512, 3, activation='relu'),
        keras.layers.Conv2D(256, 1, activation='relu'),
        keras.layers.Conv2D(512, 3, activation='relu'),
        keras.layers.Conv2D(256, 1, activation='relu'),
        keras.layers.Conv2D(512, 3, activation='relu'),
        keras.layers.Conv2D(512, 1, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu'),
        keras.layers.MaxPooling2D(2, strides=2)
    ])(layer3)

    layer5 = keras.Sequential([
        keras.layers.Conv2D(512, 1, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu'),
        keras.layers.Conv2D(512, 1, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu', strides=2),
    ])(layer4)

    layer6 = keras.Sequential([
        keras.layers.Conv2D(1024, 3, activation='relu'),
        keras.layers.Conv2D(1024, 3, activation='relu'),
    ])(layer5)

    fc1 = keras.Dense(4096)(layer6)
    fc2 = keras.Dense(30)(fc1)

    model = keras.Model(inputs=inputs, outputs=fc2)

    model.compile(optimizer=optim, loss=loss)

    return model