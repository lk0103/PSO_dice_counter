import cv2
import numpy as np
from sklearn.cluster import DBSCAN

#from Blackhat_Canny import dot_counts


class BlackHat:
    def __init__(self, img_path, blackhat_open_s=(6, 6)):
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
        Applies BlackHat morphological operation to enhance dots.
        Returns the preprocessed binary image.
        """
        img = cv2.imread(self.img_path)
        img = self.resize_image(img, 600)
        #img = cv2.medianBlur(img, 9)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (13, 13), 0)


        # BlackHat Morphology
        blackhat_img = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25)))

        # Use an opening operation to clean up noise
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.blackhat_open_s)
        open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_OPEN, se)

        # Threshold to create binary image
        _, binary_img = cv2.threshold(open_img, 50, 255, cv2.THRESH_BINARY)

        return binary_img, img

    def find_dots(self, binary_image):
        """
        Finds dots in the binary image after BlackHat and thresholding.
        """
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        dots = []
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Filter out small noise
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    dots.append((cx, cy))  # Centroid of the dot
        return dots

    def get_dice_from_dots(self, dots):
        """
        Groups blobs into dice using DBSCAN clustering.
        Returns a list of dice, each represented by the number of dots and the centroid.
        """
        X = np.asarray(dots)
        dice = []

        # DBSCAN clustering
        if dots != []:
            clustering = DBSCAN(eps=50, min_samples=1).fit(X)

            # Number of dice detected
            num_dice = max(clustering.labels_) + 1


            # Calculate the centroid for each cluster (dice) and the number of dots in each cluster
            #for i in range(num_dice):
            #    X_dice = X[clustering.labels_ == i]
            #    centroid_dice = np.mean(X_dice, axis=0)  # Average position of the dots in the cluster
            #     dice.append([len(X_dice), *centroid_dice])

            for i in range(num_dice):
                X_dice = X[clustering.labels_ == i]
                num_dots = len(X_dice)

                #if 1 <= num_dots < 7:
                centroid_dice = np.mean(X_dice, axis=0)  # Average position of the dots in the cluster
                dice.append([num_dots, *centroid_dice])

                # If the number of detected dice exceeds 6, trim to the closest 6 clusters by proximity to the centroid
            if len(dice) > 6:
                overall_centroid = np.mean([d[1:] for d in dice], axis=0)
                dice.sort(key=lambda d: np.linalg.norm(np.array(d[1:]) - overall_centroid))
                dice = dice[:6]

        return dice

    def test(self):
        binary_img, img = self.preprocess_image()
        dots = self.find_dots(binary_img)
        dice = self.get_dice_from_dots(dots)
        self.analyze_image()
        dot_counts = [0,0,0,0,0,0]
        #print(dice)
        for die in dice:

            num = die[0]-1
            #print(num, die[0])
            dot_counts[num] += 1
        return dot_counts

    def analyze_image(self):
        """
        Main function to detect dice faces and count dots using BlackHat and DBSCAN.
        """
        binary_img, img = self.preprocess_image()
        dots = self.find_dots(binary_img)

        # Clustering blobs into dice using DBSCAN
        dice = self.get_dice_from_dots(dots)


        debug_image = img.copy()  # Copy of the original image for debugging

        # Draw the blobs (dots) on the image
        for blob in dots:
            cv2.circle(debug_image, blob, 5, (0, 0, 255), -1)  # Red circles for dots

        # Draw the centroids of detected dice and the number of dots on each die
        for i, dice_info in enumerate(dice):
            num_dots, cx, cy = dice_info
            cv2.circle(debug_image, (int(cx), int(cy)), 10, (0, 255, 0), 2)  # Green circle for centroid
            cv2.putText(debug_image, f"Die {i + 1}: {num_dots} dots", (int(cx)+10, int(cy)+5 ),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Show the final result with detected dice and dots
        cv2.imshow("Detected Dice and Dots", debug_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return dice


blackhat_method = BlackHat("dataset/test_dataset/d1_1_2_3_3_5_5.jpg",(6,6))
dice_values = blackhat_method.test()
print(dice_values)
