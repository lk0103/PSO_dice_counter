import cv2
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

from Image_loader import ImageLoader

#todo upravit to na zaklade canny a potom spravit segmentaciu

class Canny:
    def __init__(self, img_path,):
        self.image_loader = ImageLoader()
        self.img_path = img_path

    def resize_image(self, image, width=800):
        height, original_width = image.shape[:2]
        aspect_ratio = height / original_width
        new_height = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (width, new_height))
        return resized_image


    def count_dots(self, processed_image):
        """
        Counts the dots based on connected components or contours.
        :param processed_image: Preprocessed binary image
        :return: List of dot counts for each die
        """
        # Find contours in the processed image
        contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on size to remove noise
        filtered_contours = [cnt for cnt in contours if 10 < cv2.contourArea(cnt) < 500]
        #print(f"Found {len(filtered_contours)} dots.")

        debug_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(debug_image, filtered_contours, -1, (0, 255, 0), 2)
        cv2.imshow("Detected Dots", debug_image)
        cv2.waitKey()


        return len(filtered_contours)

    def find_dice_with_canny(self):
        """
        Detect dice using Canny edge detection and contour analysis.
        :return: List of cropped dice sides.
        """
        # Load and resize the image
        img = cv2.imread(self.img_path)
        img = self.resize_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        # Apply Canny edge detection
        edges = cv2.Canny(blurred, threshold1=40, threshold2=100)

        # Close gaps in edges with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Detect contours
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dice_pips = []
        debug_img = img.copy()

        for contour in contours:
            # Compute area and perimeter
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
            else:
                circularity = 0

            # Filter for circular shapes
            if 10 < area < 500 and 0.6 < circularity < 1.3:
                # Mark as a dice pip
                dice_pips.append(contour)
                cv2.drawContours(debug_img, [contour], -1, (0, 255, 0), 2)

        # Show debug images
        cv2.imshow("edges",edges )
        cv2.imshow("Closed Edges", closed_edges)
        cv2.imshow("Detected Dice Pips", debug_img)
        cv2.waitKey(0)
        centroids = self.find_centroids(dice_pips)

        # Step 2: Group centroids by proximity
        groups = self.group_dots_by_proximity(centroids)

        cv2.destroyAllWindows()
        return groups

    def find_centroids(self, contours):
        """
        Calculate centroids of the detected contours.
        :param contours: List of contours (dots).
        :return: List of centroids (x, y).
        """
        centroids = []
        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroids.append((cX, cY))
        return centroids

    def group_dots_by_proximity(self, centroids, distance_threshold=70):
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
        groups = self.find_dice_with_canny()
        counted_dots = self.count_dots(groups)
        return counted_dots


img_path = "dataset/test_dataset/d16_3_4_5_5_5_5.jpg"

blackhat_processor = Canny(img_path)
#dot_counts = blackhat_processor.find_dice_with_canny()
#for dots in dot_counts:
#    cv2.imshow("dots",dots)
dot_counts = blackhat_processor.test()


print(f"Dot counts for each die: {dot_counts}")
