import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

class Images:
    def __init__(self):
        self.val_dir = "./database/validation_dataset/"
        self.val_images = self.list_files(self.val_dir)

        self.test_dir = "./database/test_dataset/"
        self.test_images = self.list_files(self.test_dir)

    def list_files(self, directory):
        files = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                files.append(filename)
        return files

    def show(self, img, title=None):
        plt.imshow(img[:, :, ::-1])
        plt.title(title)
        plt.axis('off')
        plt.show()
