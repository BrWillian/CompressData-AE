import os
from PIL import Image
import shutil


class PrepareData(object):
    def __init__(self, path):
        self._path = path
        self._images = []
        self._list_images(path)

    @property
    def images(self):
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    def separe_images_size(self, size, **kwargs):
        w, h = size
        for image in self.images:
            absolute_path = os.path.join(self._path, image)
            im = Image.open(absolute_path)
            if w < im.size[0]:
                shutil.move(absolute_path, kwargs['dst1']+image)
            else:
                shutil.move(absolute_path, kwargs['dst2']+image)






    @staticmethod
    def _list_images(path):
        tmpList = []
        for root, paths, files in os.walk(path):
            for file in files:
                tmpList.append(file)

        PrepareData.images = tmpList
