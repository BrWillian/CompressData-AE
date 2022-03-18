import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, AveragePooling2D, InputLayer
from tensorflow.python.keras.utils.layer_utils import print_summary
from model.activations import Mish

__version__ = "1.0.0"
__author__ = "brwillian"


class Encoder(Model):
    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

        # Input
        self.input_layer = InputLayer(input_shape=kwargs.get('input_shape'))

        # Block1
        self.conv_1 = Conv2D(64, (3, 3), activation=Mish(), padding='same')
        self.batch_1 = BatchNormalization()
        self.avg_polling_1 = AveragePooling2D((2, 2), padding='same')

        # Block2
        self.conv_2 = Conv2D(32, (3, 3), activation=Mish(), padding='same')
        self.batch_2 = BatchNormalization()
        self.avg_polling_2 = AveragePooling2D((2, 2), padding='same')

        # Block3
        self.conv_3 = Conv2D(16, (3, 3), activation=Mish(), padding='same')
        self.batch_3 = BatchNormalization()
        self.avg_polling_3 = AveragePooling2D((2, 2), padding='same')

    @tf.function
    def call(self, features):
        input_features = self.input_layer(features)
        x = self.conv_1(input_features)
        x = self.batch_1(x)
        x = self.avg_polling_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.avg_polling_2(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        code = self.avg_polling_3(x)

        return code


class Decoder(Model):
    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.conv_1 = Conv2D(16, (3, 3), activation=Mish(), padding='same')
        self.batch_1 = BatchNormalization()
        self.upsampling_1 = UpSampling2D((2, 2))

        # Block2
        self.conv_2 = Conv2D(32, (3, 3), activation=Mish(), padding='same')
        self.batch_2 = BatchNormalization()
        self.upsampling_2 = UpSampling2D((2, 2))

        # Block3
        self.conv_3 = Conv2D(64, (3, 3), activation=Mish(), padding='same')
        self.batch_3 = BatchNormalization()
        self.upsampling_3 = UpSampling2D((2, 2))

        self.conv_output = Conv2D(filters=kwargs.get("channels"), kernel_size=(3, 3),
                                  strides=(1, 1), activation='sigmoid', padding='same')

    @tf.function
    def call(self, features):
        x = self.conv_1(features)
        x = self.batch_1(x)
        x = self.upsampling_1(x)

        x = self.conv_2(x)
        x = self.batch_2(x)
        x = self.upsampling_2(x)

        x = self.conv_3(x)
        x = self.batch_3(x)
        x = self.upsampling_3(x)

        output = self.conv_output(x)

        return output


class CAE(Model):
    def __init__(self, **kwargs):
        super(CAE, self).__init__()
        self.encoder = Encoder(input_shape=kwargs["input_shape"])
        self.decoder = Decoder(channels=kwargs["input_shape"][-1])

    def call(self, features, **kwargs):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def summary(self, **kwargs):
        print_summary(self.encoder)
        print_summary(self.decoder)
