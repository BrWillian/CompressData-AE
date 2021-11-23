from abc import ABC
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import BatchNormalization, Conv2D, UpSampling2D, AveragePooling2D
from tensorflow.python.keras.utils.layer_utils import print_summary
from model.activations import Mish


class Autoencoder(Model, ABC):
    def __init__(self):
        super(Autoencoder, self).__init__()
        print(self._input_spec)
        self.encoder = tf.keras.Sequential([
            Conv2D(1024, (3, 3), activation=Mish(), padding='same', input_shape=(1280, 1012, 3)),
            BatchNormalization(),
            AveragePooling2D((2, 2), padding='same'),

            Conv2D(512, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            AveragePooling2D((2, 2), padding='same'),

            Conv2D(256, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            AveragePooling2D((2, 2), padding='same'),

            Conv2D(128, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            AveragePooling2D((2, 2), padding='same'),
        ],
        name="Encoder")

        self.decoder = tf.keras.Sequential([
            Conv2D(128, (3, 3), activation=Mish(), padding='same', input_shape=(80, 64, 128)),
            BatchNormalization(),
            UpSampling2D((2, 2)),

            Conv2D(256, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2)),

            Conv2D(512, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2)),

            Conv2D(1024, (3, 3), activation=Mish(), padding='same'),
            BatchNormalization(),
            UpSampling2D((2, 2)),

            Conv2D(3, (3, 3), activation='sigmoid', padding='same')
        ],
        name="Decoder")

    @tf.function
    def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

    def summary(self, **kwargs):
        print_summary(self.encoder)
        print_summary(self.decoder)