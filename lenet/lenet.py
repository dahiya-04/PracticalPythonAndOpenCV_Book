from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dense
from tensorflow.keras import backend as K

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)

        # Adjust if channels first format is used
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

        # First CONV => RELU => POOL layer
        model.add(Conv2D(20, (5, 5), padding='same', input_shape=inputShape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Second CONV => RELU => POOL layer
        model.add(Conv2D(20, (5, 5), padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Flatten and fully connected layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation('relu'))

        # Output layer
        model.add(Dense(classes))
        model.add(Activation('softmax'))

        return model
