import cv2
import numpy as np
import os


class DatasetGenerator:
    def __init__(self, image_path, batch_size=32, size=(1280, 1012, 3)):
        self.image_path = image_path
        self.batch_size = batch_size
        self.size = (size[0], size[1])
        self.shape = "({}, {}, {}, {})".format(batch_size, size[0], size[1], size[2])

    @staticmethod
    def list_dir(path):
        data = []
        for _, _, files in os.walk(path):
            for file in files:
                data.append(file)

        data = list(map(lambda s: s.split('.')[0], data))
        return data

    def create_dataset(self):
        data = DatasetGenerator.list_dir(self.image_path)
        while True:
            for start in range(0, len(data), self.batch_size):
                x_batch = []
                end = min(start + self.batch_size, len(data))
                id_train_batch = data[start:end]
                for id in id_train_batch:
                    img = cv2.imread(self.image_path + '{}.jpg'.format(id), cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
                    x_batch.append(img)

                x_batch = np.array(x_batch, np.float32) / 255
                yield x_batch
