import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import os


def view_dataset_representation(img_folder, n_samples):
    plt.figure(figsize=(20, 20))
    for i in range(n_samples):
        file = random.choice(os.listdir(img_folder))
        image_path = os.path.join(img_folder, file)
        img = mpimg.imread(image_path)
        plt.imshow(img)
        plt.show()