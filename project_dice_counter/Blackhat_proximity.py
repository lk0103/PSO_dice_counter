import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from Image_loader import ImageLoader


#todo treba pridat pocitanie bodiek v skupine

class BlackHat:
    def __init__(self, img_path,blackhat_open_s=(6, 6) ):
        self.image_loader = ImageLoader()
        self.img_path = img_path
        self.blackhat_open_s = blackhat_open_s

    def resize_image(self, image, width=800):
        height, original_width = image.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        return resized_image

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
        cv2.imshow("BlackHat Image 1", blackhat_img)
        cv2.waitKey()

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.blackhat_open_s)
        open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_OPEN, se)
        cv2.imshow("BlackHat Image", blackhat_img)
        cv2.waitKey()

        # Threshold the blackhat image
        _, binary_img = cv2.threshold(open_img, 50, 100, cv2.THRESH_BINARY)
        cv2.imshow("Thresholded BlackHat Image", binary_img)
        cv2.waitKey()

        return binary_img
    def find_dot_centroids(self):
        """
        Finds centroids of detected dots from the preprocessed image.
        :param processed_image: Binary image after blackhat preprocessing.
        :return: List of centroids (x, y) of the detected dots.
        """
        processed_image = self.preprocess_image()
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []

        for cnt in contours:
            if 10 < cv2.contourArea(cnt) < 500:  # Filter out noise
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centroids.append((cX, cY))

        # Debug: Show detected centroids
        debug_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        for c in centroids:
            cv2.circle(debug_image, c, 5, (0, 0, 255), -1)
        cv2.imshow("Detected Centroids", debug_image)
        cv2.waitKey(0)

        return centroids

    def group_dots_by_proximity(self, centroids, distance_threshold=50):
        """
        Groups dots based on their proximity and validates the group size.
        :param centroids: List of centroids (x, y) of detected dots.
        :param distance_threshold: Maximum distance between dots to consider them part of the same group.
        :return: List of valid groups (1 to 6 centroids each).
        """
        groups = []
        used = set()

        for i, (x1, y1) in enumerate(centroids):
            if i in used:
                continue
            group = [(x1, y1)]
            used.add(i)

            for j, (x2, y2) in enumerate(centroids):
                if j not in used:
                    dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                    if dist < distance_threshold:
                        group.append((x2, y2))
                        used.add(j)

            # Validate group size (1 to 6 dots)
            if 1 <= len(group) <= 6:
                groups.append(group)
        img = cv2.imread(self.img_path)
        # Debug: Visualize grouped dots
        debug_image = np.zeros((2000, 2000, 3), dtype=np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for i, group in enumerate(groups):
            color = colors[i % len(colors)]
            for c in group:
                cv2.circle(debug_image, c, 5, color, -1)
        cv2.imshow("Valid Groups", debug_image)
        cv2.waitKey(0)

        return groups

    def count_dots(self,groups):
        counted_dots = [0,0,0,0,0,0]

        for i in groups:
            counted_dots[len(i)-1] += 1
        return counted_dots

    def test(self):
        centroids = self.find_dot_centroids()
        groups = self.group_dots_by_proximity(centroids)
        counted_dots = self.count_dots(groups)
        return counted_dots

    def estimate_die_orientation(self, groups):
        """
        Estimates the orientation and bounding boxes of dice based on grouped dots.
        :param groups: List of groups, each containing centroids of one die.
        :return: List of bounding boxes (rotated rectangles) for each die.
        """
        bounding_boxes = []

        for group in groups:
            points = np.array(group, dtype=np.int32)
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect)
            #box = np.int0(box)
            bounding_boxes.append(box)

        # Debug: Visualize bounding boxes
        debug_image = np.zeros((500, 500, 3), dtype=np.uint8)
        for box in bounding_boxes:
            cv2.drawContours(debug_image, [box], 0, (0, 255, 0), 2)
        cv2.imshow("Bounding Boxes", debug_image)
        cv2.waitKey(0)

        return bounding_boxes

img_path = "dataset/test_dataset/d3_1_2_4_6_6_6.jpg"
blackhat_processor = BlackHat(img_path)


img_path = "dataset/test_dataset/d1_1_2_3_3_5_5.jpg"
blackhat_processor = BlackHat(img_path)
#dot_counts = blackhat_processor.find_dice_with_canny()
#for dots in dot_counts:
#    cv2.imshow("dots",dots)
centroids = blackhat_processor.find_dot_centroids()
groups = blackhat_processor.group_dots_by_proximity(centroids)
counted_dots = blackhat_processor.count_dots(groups)
#print(counted_dots)


#print(f"Dot counts for each die: {dot_counts}")