import prepare_data
from model.model import CAE
from prepare_data import DatasetGenerator
from view_dataset import view_dataset_representation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import mse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


dataset_folder = "/home/willian/images/high/"

#prepare = prepare_data.PrepareData("/media/Shared Disk/Dataset/Imagens-Labor")
#prepare.separe_images_size((1270, 1012), dst1="/media/Shared Disk/Dataset/high/",
                           #dst2="/media/Shared Disk/Dataset/mid/")



#view_dataset_representation(dataset_folder, n_samples=10)

mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])

with mirrored_strategy.scope():
    model = CAE(input_shape=(28, 28, 3))
    model.summary()


#Datagen = DatasetGenerator(dataset_folder)
#Dataset = Datagen.create_dataset()
#print(Datagen.shape)

train_datagen = ImageDataGenerator(rescale=1 / 255)
train_generator = train_datagen.flow_from_directory('/home/willian/images/', target_size=(28, 28),
                  color_mode='rgb', class_mode='input', batch_size=32)

model.compile(optimizer=Adam(learning_rate=0.00025), loss="binary_crossentropy"
              )


model.fit(train_generator, epochs=30)