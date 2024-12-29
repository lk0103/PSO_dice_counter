import cv2
import numpy as np

from Image_loader import ImageLoader

class KmeansSegmentation():
    def segment_k_means(self, img, k, conversion=None, reverse_conversion=None):
        h, w = img.shape[0], img.shape[1]

        if conversion is not None:
            img = cv2.cvtColor(img, conversion).astype(np.float32)
        else:
            img = img.astype(np.float32)

        pixels = img.reshape((h * w, -1))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # Reshape the labels to match the original image shape
        segmentation = labels.reshape((h, w))

        posterized_img = np.floor(centers[segmentation]).astype(np.uint8)

        if reverse_conversion is not None:
            posterized_img = cv2.cvtColor(posterized_img, reverse_conversion)

        if self.has_one_channels(posterized_img):
            posterized_img = posterized_img.reshape((h, w))
        return posterized_img, centers

    def has_one_channels(self, img):
        return len(img.shape) > 2 and img.shape[2] == 1

    def has_three_channels(self, img):
        return len(img.shape) > 2 and img.shape[2]  == 3

    def unsharp_mask(self, img, p, sigma):
        img_blurred = cv2.GaussianBlur(img, (15, 15), sigma, sigma)
        return img - p * (img - img_blurred)

    def segment_k_means_masked(self, mask, img, k, conversion=None, reverse_conversion=None):

        if conversion is not None and self.has_three_channels(img):
            img = cv2.cvtColor(img, conversion)
            img = self.unsharp_mask(img, 0.01, 5).astype(np.float32)
            img = cv2.GaussianBlur(img, (5, 5), 0)
        else:
            img = img.astype(np.float32)

        pixels = img[mask]


        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)
        labels = labels.flatten()
        centers = centers.flatten()

        posterized_img = np.floor(centers[labels]).astype(np.uint8).flatten()


        new_img = np.zeros_like(mask, dtype=np.uint8)
        new_img[mask] = np.floor(posterized_img).astype(np.uint8)
        new_img[~mask] = np.floor(img[~mask]).astype(np.uint8)
        if reverse_conversion is not None and self.has_three_channels(img):
            new_img = cv2.cvtColor(new_img, reverse_conversion)

        return new_img, centers


    def find_closest_to_black_mask(self, posterized_img, centers):
        black_lab = np.array([0, 128, 128], dtype=np.float32)  # Black in LAB color space

        distances = np.linalg.norm(centers - black_lab, axis=1)
        closest_to_black_idx = np.argmin(distances)

        posterized_img = cv2.cvtColor(posterized_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        mask = np.all(np.abs(posterized_img - np.floor(centers[closest_to_black_idx]) <= 1).astype(np.uint8), axis=-1)
        mask_values = mask.astype(np.uint8) * 255

        return mask, mask_values

    def find_closest_to_black_mask_grayscale(self, posterized_img):
        darkest_value = np.min(posterized_img)

        mask = (posterized_img == darkest_value)
        mask_values = mask.astype(np.uint8) * 255


        return mask, mask_values

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
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

        # Apply K-means clustering
        _, labels, centers = cv2.kmeans(centroids, n_clusters, None, criteria, 3, cv2.KMEANS_RANDOM_CENTERS)

        # Count the number of contours in each cluster
        cluster_counts = np.bincount(labels.flatten(), minlength=n_clusters)

        # Sort the cluster counts in ascending order
        sorted_counts = list(map(int, sorted(cluster_counts)))
        sorted_counts = [min(int(x), 6) for x in sorted(cluster_counts)]

        return sorted_counts

    def black_hat_dataset(self):
        images_class = ImageLoader()

        mistakes = 0

        for img_path in images_class.val_images:
            # if img_path[-19:-16] not in ('d12', 'd19'): #('d12', 'd17', 'd24'):
            #     continue
            print(img_path)
            img = cv2.imread(img_path)

            img = cv2.medianBlur(img, 41)
            # images_class.show(img, title='original')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #images_class.show(img, title='original')

            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
            blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)
            #images_class.show(blackhat_img, title='original')

            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))
            open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_OPEN, se)
            #images_class.show(open_img, title='opened')

            #bin_img = cv2.adaptiveThreshold(open_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 2)
            _, bin_img = cv2.threshold(open_img, 10, 255, cv2.THRESH_BINARY)
            #images_class.show(bin_img, title='binarized')
            #iba obrazok 6 a 8 nemaju vsetky bodky

            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (28, 28))
            open_img = cv2.morphologyEx(bin_img, cv2.MORPH_ERODE, se)
            #images_class.show(open_img, title='eroded')

            contours, num_contours = self.find_and_filter_contours(open_img)

            rolls, rolls_sum = images_class.extract_dice_rolls(img_path)

            mistakes += abs(num_contours - rolls_sum)

            print(f'number of contours = {num_contours}, correct roll sum = {rolls_sum}')

            counts = self.cluster_contours_and_count_cv2(contours)
            print("Ordered contour counts by cluster:", counts, ' correct rolls: ', rolls)

        print(f'\n number of mistakes in total number of cicles = {mistakes}')

    def find_and_filter_contours(self, img):
        images_class = ImageLoader()

        cnt, h = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # images_class.show(img, 'Image with {} circles'.format(len(cnt)))
        filtered_contours = self.filter_circles_by_shape_and_area(cnt, min_circularity=0.7, area_tolerance=1.5)

        output_img = np.zeros_like(img)
        output_img = cv2.drawContours(output_img, filtered_contours, -1, 255, thickness=cv2.FILLED)
        # Display results
        #images_class.show(output_img, 'Filtered Image with {} circles'.format(len(filtered_contours)))

        return filtered_contours, len(filtered_contours)

    def posterize_dataset(self):
        images_class = ImageLoader()

        for img_path in images_class.val_images:
            # if img_path[-19:-16] not in ('d12', 'd19'): #('d12', 'd17', 'd24'):
            #     continue
            print(img_path)
            img = cv2.imread(img_path)

            img = cv2.medianBlur(img, 41)
            #images_class.show(img, title='original')

            gray = False
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                posterized_img, centers = self.segment_k_means(img, 3)
                # images_class.show(posterized_img, title='segmented')

                mask, mask_darkest = self.find_closest_to_black_mask_grayscale(posterized_img)
                # images_class.show_gray(mask_darkest, title='masked')
            else:
                posterized_img, centers = self.segment_k_means(img, 3, cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR)
                # images_class.show(posterized_img, title='segmented')

                mask, mask_darkest = self.find_closest_to_black_mask(posterized_img, centers)
                # images_class.show_gray(mask_darkest, title='masked')


            masked_img = np.full_like(posterized_img, 255)
            masked_img[mask] = img[mask]
            if gray:
                images_class.show_gray(masked_img, title='segmented')
            else:
                images_class.show(masked_img, title='segmented')


            posterized_masked, centers = self.segment_k_means_masked(mask=mask, img=masked_img, k=7,
                                                                     conversion=cv2.COLOR_BGR2GRAY,
                                                                     reverse_conversion=cv2.COLOR_GRAY2BGR)
            #images_class.show(posterized_masked, title='second kmeans masked')

            mask, mask_darkest = self.find_closest_to_black_mask_grayscale(posterized_masked)
            images_class.show_gray(mask_darkest, title='second kmeans darkest color')

            se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (48, 48))

            tophat_img = cv2.morphologyEx(mask_darkest, cv2.MORPH_TOPHAT, se)


            #a odstranit contoury, ktore maju obsah vacsi ako nejaka hodnota !!!!!!!!!!!!!!!!
            #!!!!!!!!!!!!!!!!!!!!!!

            # se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            # changed = cv2.morphologyEx(tophat_img, cv2.MORPH_CLOSE, se)
            images_class.show_gray(tophat_img, title='changed')

            # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            # mask_darkest = cv2.morphologyEx(mask_darkest, cv2.MORPH_OPEN, kernel, iterations=2)
            # #opened_img = cv2.morphologyEx(opened_img, cv2.MORPH_ERODE, kernel)
            # images_class.show_gray(mask_darkest, title='masked')

            # hit_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
            # hit_kernel = np.where(hit_kernel == 0, -1, 1)
            # hit_result = cv2.morphologyEx(opened_img, cv2.MORPH_HITMISS, hit_kernel)
            # images_class.show_gray(hit_result, title='masked')

            # hit_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            # hit_result = cv2.morphologyEx(hit_result, cv2.MORPH_HITMISS, hit_kernel)
            # images_class.show_gray(hit_result, title='masked')

            #images_class.show_gray(hit_result, title='masked')

    def masking(self, images_class, img, mask):
        masked_img = np.zeros_like(img)
        masked_img[mask] = img[mask]
        # images_class.show(masked_img, title='segmented')
        masked_img = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        adaptive_thresh = cv2.adaptiveThreshold(
            masked_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        adaptive_thresh = 255 - adaptive_thresh
        images_class.show_gray(adaptive_thresh, title='segmented')
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30))  # Circular kernel (adjust size)
        # Closing operation (fill the circles)
        closed = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        images_class.show_gray(closed, title='segmented')
        opened_image = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        images_class.show_gray(opened_image, title='segmented')


# KmeansSegmentation().posterize_dataset()
KmeansSegmentation().black_hat_dataset()