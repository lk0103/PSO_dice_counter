import cv2
import numpy as np

from Image_loader import ImageLoader
from ContourOperations import ContourOperations

class KmeansSegmentation():
    def __init__(self, img_path, show=False, point_color='b',
                 rgb_k=3, gray_k=7, median_blur_s=41,
                 blackhat_open_s=(30, 30), open_s=(10, 10)):
        self.image_loader = ImageLoader()
        self.img_path = img_path
        self.show = show
        self.rgb_k = rgb_k
        self.gray_k = gray_k
        self.median_blur_s = median_blur_s
        self.blackhat_open_s = blackhat_open_s
        self.open_s = open_s

        if point_color == 'b':
            self.find_darkest = True
            self.goal_color_lab = np.array([0, 128, 128], dtype=np.float32)  # Black in LAB color space
        else:
            self.find_darkest = False
            self.goal_color_lab = np.array([100, 0, 0], dtype=np.float32)  # White in LAB color space


    def find_dice_rolls_with_kmeans(self):
        open_img = self.kmeans_find_dice_points()

        contours = ContourOperations(show=self.show)
        dice_rolls, num_contours = contours.count_contours_cluster_to_6_dices(open_img)
        return dice_rolls, num_contours

    def kmeans_find_dice_points(self):
        print(self.img_path)
        img = self.median_blur_image()
        # if self.show:
        #     self.image_loader.show(img, title='median blur')

        binary_mask_darkest, rgb_mask_darkest = self.kmeans_rgb_colors_find_darkest(img)

        mask_darkest = self.kmeans_grayscale_find_darkest(binary_mask_darkest, rgb_mask_darkest)

        cond_dilated_img = self.conditional_dilation(mask_darkest)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.open_s)
        open_img = cv2.morphologyEx(cond_dilated_img, cv2.MORPH_OPEN, se)
        # if self.show:
        #     self.image_loader.show(open_img, title='opened')

        return open_img

    def k_means_colors(self, img, k, conversion=None, reverse_conversion=None):
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
        return len(img.shape) > 2 and img.shape[2] == 3

    def unsharp_mask(self, img, p, sigma):
        img_blurred = cv2.GaussianBlur(img, (15, 15), sigma, sigma)
        return img - p * (img - img_blurred)

    def k_means_colors_masked_image(self, mask, img, k, conversion=None, reverse_conversion=None):

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
        distances = np.linalg.norm(centers - self.goal_color_lab, axis=1)
        closest_to_black_idx = np.argmin(distances)

        posterized_img = cv2.cvtColor(posterized_img, cv2.COLOR_BGR2Lab).astype(np.float32)
        mask = np.all(np.abs(posterized_img - np.floor(centers[closest_to_black_idx]) <= 1).astype(np.uint8), axis=-1)
        mask_values = mask.astype(np.uint8) * 255

        return mask, mask_values

    def find_closest_to_black_mask_grayscale(self, posterized_img):
        if self.find_darkest:
            #darkest_value = np.min(posterized_img)
            darkest_value = np.sort(np.unique(posterized_img.flatten()))[:2]
        else:
            #darkest_value = np.max(posterized_img)
            darkest_value = np.sort(np.unique(posterized_img.flatten()))[-2:]

        #mask = (posterized_img == darkest_value)
        mask = np.isin(posterized_img, darkest_value)
        mask_values = mask.astype(np.uint8) * 255

        return mask_values

    def median_blur_image(self):
        img = cv2.imread(self.img_path)
        img = cv2.medianBlur(img, self.median_blur_s)
        return img

    def kmeans_rgb_colors_find_darkest(self, img):
        posterized_img, centers = self.k_means_colors(img, self.rgb_k, cv2.COLOR_BGR2Lab, cv2.COLOR_Lab2BGR)
        # odkomentovat
        # if self.show:
        #     self.image_loader.show(posterized_img, title='posterized image kmeans')

        binary_mask_darkest, mask_darkest = self.find_closest_to_black_mask(posterized_img, centers)
        #self.image_loader.show_gray(mask_darkest, title='masked darkest color')

        rgb_mask_darkest = np.full_like(posterized_img, 255)
        rgb_mask_darkest[binary_mask_darkest] = img[binary_mask_darkest]
        # if self.show:
        #     self.image_loader.show(rgb_mask_darkest, title='mask darkest color - in original colors')
        return binary_mask_darkest, rgb_mask_darkest

    def kmeans_grayscale_find_darkest(self, binary_mask_darkest, rgb_mask_darkest):
        posterized_masked, centers = self.k_means_colors_masked_image(mask=binary_mask_darkest,
                                img=rgb_mask_darkest, k=self.gray_k, conversion=cv2.COLOR_BGR2GRAY,
                                reverse_conversion=cv2.COLOR_GRAY2BGR)
        # # odkomentovat
        # if self.show:
        #     self.image_loader.show(posterized_masked, title='grayscale kmeans on masked image')
        mask_darkest = self.find_closest_to_black_mask_grayscale(posterized_masked)
        # if self.show:
        #     self.image_loader.show_gray(mask_darkest, title='grayscale kmeans darkest color')
        return mask_darkest

    def conditional_dilation(self, mask_darkest):
        blackhat_img = self.blackhat_gray_image()
        blackhat_img = (blackhat_img > 0).astype(np.uint8)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20))
        img_dil = cv2.morphologyEx(mask_darkest, cv2.MORPH_DILATE, se)

        conditional = img_dil * blackhat_img

        final_img = np.where(mask_darkest == 0, conditional, mask_darkest)
        # odkomentovat
        # if self.show:
        #     self.image_loader.show(final_img, title='conditional')
        return final_img

    def blackhat_gray_image(self):
        img = cv2.imread(self.img_path)
        img = cv2.medianBlur(img, self.median_blur_s)
        # self.image_loader.show(img, title='original')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # self.image_loader.show(img, title='original')
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (55, 55))
        blackhat_img = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, se)

        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.blackhat_open_s)
        open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_OPEN, se)
        # self.image_loader.show(open_img, title='opened')

        _, bin_img = cv2.threshold(open_img, 10, 255, cv2.THRESH_BINARY)

        #odkomentovat
        # if self.show:
        #     self.image_loader.show(bin_img, title='blackhat image used for conditional')
        return bin_img


    #toto samo o sebe tiez celkom dobre extrahuje points
    # def blackhat_point_extraction(self):
    #     blackhat_img = self.blackhat_gray_image()
    #     # self.image_loader.show(blackhat_img, title='original')
    #
    #     se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (28, 28))
    #     open_img = cv2.morphologyEx(blackhat_img, cv2.MORPH_ERODE, se)
    #     # self.image_loader.show(open_img, title='eroded')
    #     return open_img
