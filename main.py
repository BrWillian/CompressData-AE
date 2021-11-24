import prepare_data
from model.model import Autoencoder
from prepare_data import DatasetGenerator
from view_dataset import view_dataset_representation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import numpy as np
import tensorflow as tf


dataset_folder = "/media/Shared Disk/Dataset/high/"

#prepare = prepare_data.PrepareData("/media/Shared Disk/Dataset/Imagens-Labor")
#prepare.separe_images_size((1270, 1012), dst1="/media/Shared Disk/Dataset/high/",
                           #dst2="/media/Shared Disk/Dataset/mid/")



#view_dataset_representation(dataset_folder, n_samples=10)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with mirrored_strategy.scope():
    model = Autoencoder()
    model.build(input_shape=(None, 1280, 1012, 3))
    model.summary()


Datagen = DatasetGenerator(dataset_folder)
Dataset = Datagen.create_dataset()
print(Datagen.shape)
model.compile(optimizer=Adam(learning_rate=0.00025), loss="binary_crossentropy",
              metrics=["loss"])


model.fit(Dataset, Dataset, batch_size=32, epochs=30)