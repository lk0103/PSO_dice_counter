import cv2
import numpy as np
from sklearn.cluster import DBSCAN

from Image_loader import ImageLoader

class ContourOperations():
    def __init__(self, show = False):
        self.image_loader = ImageLoader()
        self.show = show

    def count_contours_cluster_to_6_dices(self, img):
        contours, num_contours, final_image = self.find_and_filter_contours(img)

        try:
            dice_rolls, centers = self.cluster_contours_and_count_cv2(contours)
        except:
            print(f"Not enough contours to form 6 clusters.")
            dice_rolls = [0, 0, 0, 0, 0, 0]
            centers = []

        self.show_image_with_cluster_centers(centers, final_image)
        return dice_rolls, num_contours

    def find_and_filter_contours(self, img):
        cnt, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = self.filter_circles_by_shape_and_area(cnt, min_circularity=0.85, area_tolerance=1.)

        output_img = np.zeros_like(img)
        output_img = cv2.drawContours(output_img, filtered_contours, -1, 255, thickness=cv2.FILLED)

        return filtered_contours, len(filtered_contours), output_img

    def filter_circles_by_shape_and_area(self, contours, min_circularity=0.7, area_tolerance=0.2):
        valid_contours = []
        areas = []

        # Calculate areas of all contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 0:
                areas.append(area)

        # Compute average area
        avg_area = np.mean(areas) if areas else 0
        min_area = (1 - area_tolerance) * avg_area
        max_area = (1 + area_tolerance) * avg_area

        # Filter contours
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:  # Avoid division by zero
                continue

            # Calculate circularity
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            if circularity >= min_circularity:
                valid_contours.append(cnt)

        return valid_contours

    def cluster_contours_and_count_cv2(self, contours, n_clusters=6):
        # Extract the centroid of each contour
        centroids = []
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] != 0:  # Avoid division by zero
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append([cx, cy])

        if len(centroids) < n_clusters:
            raise ValueError(f"Not enough contours ({len(centroids)}) to form {n_clusters} clusters.")

        # Convert centroids to numpy array
        centroids = np.array(centroids, dtype=np.float32)

        # Define K-means criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.05)

        # Apply K-means clustering
        _, labels, centers = cv2.kmeans(centroids, n_clusters, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)

        # Count the number of contours in each cluster
        cluster_counts = np.bincount(labels.flatten(), minlength=n_clusters)

        # Sort the cluster counts in ascending order
        #sorted_counts = list(map(int, sorted(cluster_counts)))
        sorted_counts = [min(int(x), 6) for x in sorted(cluster_counts)]

        return sorted_counts, centers

    def show_image_with_cluster_centers(self, centers, final_image):
        final_image = cv2.cvtColor(final_image, cv2.COLOR_GRAY2BGR)
        for center in centers:
            center = tuple(map(int, center))  # Convert to integer coordinates
            cv2.circle(final_image, center, radius=10, color=(0, 0, 255), thickness=-1)  # Red filled circle
        # Show the image with centers drawn
        if self.show:
            self.image_loader.show(final_image, title='Centers of clusters')