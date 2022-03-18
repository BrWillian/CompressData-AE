import cv2
import numpy as np
import os


class DatasetGenerator:
    def __init__(self, image_path, batch_size=32, size=(28, 28, 3)):
        self.image_path = image_path
        self.batch_size = batch_size
        self.size = (size[0], size[1])
        self.shape = "({}, {}, {}, {})".format(batch_size, size[0], size[1], size[2])
        self.len_data = 0

    @staticmethod
    def list_dir(path):
        data = []
        for _, _, files in os.walk(path):
            for file in files:
                data.append(file)

        data = list(map(lambda s: s.split('.')[0], data))
        return data

    def create_dataset(self):
        img_data_array = []
        img_folder = self.image_path

        for root, dirs, files in os.walk(img_folder):
            for file in files:
                image_path = os.path.join(root, file)
                image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
                image = np.array(image)
                image = image.astype('float32')
                image /= 255
                img_data_array.append(image)
        return img_data_array

    def create_dataset_autogen(self):
        data = DatasetGenerator.list_dir(self.image_path)
        while True:
            for start in range(0, len(data), self.batch_size):
                x_batch = []
                y_batch = []
                end = min(start + self.batch_size, len(data))
                id_train_batch = data[start:end]
                for id in id_train_batch:
                    x = cv2.imread(self.image_path + '{}.jpg'.format(id))
                    x = cv2.resize(x, (28, 28))
                    y = cv2.imread(self.image_path + '{}.jpg'.format(id), cv2.IMREAD_GRAYSCALE)
                    y = cv2.resize(y, (28, 28))
                    y = np.expand_dims(y, axis=2)
                    print(x.shape)
                    x_batch.append(x)
                    y_batch.append(y)

                x_batch = np.array(x_batch, np.float32) / 255
                y_batch = np.array(y_batch, np.float32) / 255
                yield x_batch, y_batch
