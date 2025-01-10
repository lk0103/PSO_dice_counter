import cv2
import numpy as np
from matplotlib import pyplot as plt


class TemplateDotMatch:
    def __init__(self, img_path, size,blackhat_open_s=(6, 6)):
        self.img_path = img_path
        self.dot_template = self.create_dot_template(size)
        self.blackhat_open_s = blackhat_open_s


    def resize_image(self, image, width=800):
        height, original_width = image.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        return resized_image

    def create_dot_template(self, size=15):
        """
        Creates a circular dot template to be used for template matching.
        :param size: Diameter of the dot in pixels.
        :return: A grayscale image with a filled circle.
        """
        template = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = size // 2 - 1  # Slightly smaller than the size to fit within the bounds
        cv2.circle(template, center, radius, 200, -1)  # Draw filled circle
        return template

    def preprocess_image(self):
        """
        Loads, blurs, and thresholds the input image to isolate the dots.
        :return: Preprocessed binary image.
        """
        img = cv2.imread(self.img_path, cv2.IMREAD_GRAYSCALE)
        img = self.resize_image(img)
        if img is None:
            raise ValueError(f"Image at path {self.img_path} could not be loaded.")

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        cv2.imshow("", blurred)
        cv2.waitKey(0)
        # Apply threshold to isolate dark regions (dots)
        _, thresholded = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("", thresholded)
        cv2.waitKey(0)
        return thresholded

    def preprocess_image(self):
        """
        Applies blackhat morphological operation to enhance the dots.
        :return: Processed binary image
        """

        img = cv2.imread(self.img_path)
        img = self.resize_image(img,600)
        #cv2.imshow('classic',img)
        #cv2.waitKey(0)
        #img = cv2.medianBlur(img, 11)
        #cv2.imshow('blur',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.GaussianBlur(img, (11,11), 0)



        blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25,25)))
        #cv2.imshow("BlackHat Image", blackhat_img)
        #cv2.waitKey()

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.blackhat_open_s)
        open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_OPEN, se)
        #cv2.imshow("BlackHat Image", blackhat_img)
        #cv2.waitKey()

        # Threshold the blackhat image
        _, binary_img = cv2.threshold(open_img, 50, 100, cv2.THRESH_BINARY)
        cv2.imshow("Thresholded BlackHat Image", binary_img)
        cv2.waitKey()

        return binary_img

    def match_dots(self):
        """
        Matches the dot template to the preprocessed image to detect dots.
        :return: List of dot locations (x, y) and the original image with dots marked.
        """
        # Preprocess the input image
        preprocessed_img = self.preprocess_image()
        preprocessed_img = self.resize_image(preprocessed_img)

        # Perform template matching
        result = cv2.matchTemplate(preprocessed_img, self.dot_template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.1  # Confidence threshold for matching
        locations = np.where(result >= threshold)

        # Original image for visualization
        orig_img = cv2.imread(self.img_path)
        orig_img = self.resize_image(orig_img)
        if orig_img is None:
            raise ValueError(f"Image at path {self.img_path} could not be loaded.")

        h, w = self.dot_template.shape
        dot_locations = []

        for pt in zip(*locations[::-1]):
            dot_locations.append(pt)
            # Draw a rectangle around each detected dot
            center = (pt[0] + 10, pt[1] +10)
            cv2.circle(orig_img,center , 3, (0, 255, 0), 0)

        return dot_locations, orig_img

    def visualize_dots(self):
        """
        Visualizes the detected dots on the original image.
        """
        dot_locations, marked_img = self.match_dots()
        marked_img = self.resize_image(marked_img)
        # Show the image with marked dots
        cv2.imshow("Detected Dots", marked_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #print(f"Detected dots at locations: {dot_locations}")


img_path = "dataset/test_dataset/d3_1_2_4_6_6_6.jpg"
temp = TemplateDotMatch(img_path,20)
temp.visualize_dots()
#for i in range(20,100,5):
#    temp = TemplateDotMatch(img_path,i)
#    temp.visualize_dots()
