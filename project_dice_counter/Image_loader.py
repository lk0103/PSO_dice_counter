import os
import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('TkAgg')

class ImageLoader():
    def __init__(self):
        self.val_dir = "./dataset/validation_dataset/"
        self.val_images = self.list_files(self.val_dir)

        self.test_dir = "./dataset/test_dataset/"
        self.test_images = self.list_files(self.test_dir)

    def list_files(self, directory):
        files = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                files.append(directory + filename)
        return files

    def show(self, img, title=None):
        if self.is_grayscale(img):
            self.show_gray(img, title)
            return
        plt.imshow(img[:, :, ::-1])
        plt.title(title)
        plt.axis('off')
        plt.show()

    def show_gray(self, img, title=None):
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

    def is_grayscale(self, img):
        return len(img.shape) == 2

    def extract_dice_rolls_from_filename(self, file_path):
        file_name = os.path.basename(file_path)

        name_without_ext = os.path.splitext(file_name)[0]

        rolls = list(map(int, name_without_ext.split('_')[-6:]))
        rolls_sum = sum(rolls)

        return rolls, rolls_sum




