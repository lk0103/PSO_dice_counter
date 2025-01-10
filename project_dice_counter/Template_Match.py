import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class TemplateMatch:
    def __init__(self, img_path):
        self.img_path = img_path
        self.templates = self.get_templates()

    def resize_image(self, image, width=800):
        height, original_width = image.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        return resized_image

    def get_templates(self):
        """Load all dice templates from the dataset/templates/ directory."""
        path = "dataset/sides/"
        templates = {}
        for i in range(1, 7):
            templates[i] = cv2.imread(path + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
            templates[i] = templates[i].astype(np.uint8)

        return templates

    def template_match(self):
        """Perform template matching to count the number of dots on the dice."""
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        img = self.resize_image(img)
        img = img.astype(np.uint8)
        cv2.imshow('img', img)
        cv2.waitKey(0)
        if img is None:
            raise ValueError(f"Image at path {self.img_path} could not be loaded.")

        # Placeholder for detected dots count
        dot_counts = {}

        # Loop through each template and perform matching
        for side_name, template in self.templates.items():
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5  # Adjust based on experimentation
            locations = np.where(result >= threshold)
            dot_counts[side_name] = len(list(zip(*locations[::-1])))

        return dot_counts

    def visualize_matches(self):
        """Visualize template matching results for debugging."""
        img = cv2.imread(self.img_path,cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Image at path {self.img_path} could not be loaded.")

        for side_name, template in self.templates.items():
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            threshold = 0.5
            locations = np.where(result >= threshold)
            h, w = template.shape

            for pt in zip(*locations[::-1]):
                cv2.rectangle(img, pt, (pt[0] , pt[1]), (0, 0, 255), 2)

        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title("Template Matches")
        plt.show()


img_path = "dataset/test_dataset/d3_1_2_4_6_6_6.jpg"
temp = TemplateMatch(img_path)
temp.template_match()
temp.visualize_matches()