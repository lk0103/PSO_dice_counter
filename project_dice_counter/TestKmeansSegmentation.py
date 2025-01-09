import os

from Image_loader import ImageLoader
from Kmeans_segmentation import KmeansSegmentation

class TestKmeansSegmentation():
    def __init__(self):
        self.image_loader = ImageLoader()

    def compare_dice_rolls(self, found_rolls, correct_rolls):

        # Measure 1: Count of incorrectly identified dice rolls
        incorrect_rolls = sum(1 for f, c in zip(found_rolls, correct_rolls) if f != c)

        # Measure 2: Total difference in dice rolls
        roll_difference = sum(abs(f - c) for f, c in zip(found_rolls, correct_rolls))

        return incorrect_rolls, roll_difference

    # def test_kmeans_dataset(self, image_dataset='val'):
    #     count_diff_point_sum = 0
    #     incorrect_rolls_total = 0
    #     roll_difference_total = 0
    #
    #     dataset = self.set_dataset(image_dataset)
    #
    #     for img_path in dataset:
    #         kmeans_seg = KmeansSegmentation(img_path, False)
    #         extracted_dice_rolls, num_contours = kmeans_seg.find_dice_rolls_with_kmeans()
    #
    #         rolls, rolls_sum = self.image_loader.extract_dice_rolls_from_filename(img_path)
    #
    #         count_diff_point_sum += abs(num_contours - rolls_sum)
    #
    #         print(f'number of contours = {num_contours}, correct roll sum = {rolls_sum}')
    #
    #         print("found rolls:", extracted_dice_rolls, ' correct rolls: ', rolls)
    #
    #         incorrect_rolls, roll_difference = self.compare_dice_rolls(extracted_dice_rolls, rolls)
    #         incorrect_rolls_total += incorrect_rolls
    #         roll_difference_total += roll_difference
    #         print(f"Incorrect dice rolls: {incorrect_rolls}, Difference of rolls: {roll_difference}\n")
    #
    #
    #     print(f'\n total difference between segmented dice points and correct number of point = {count_diff_point_sum}')
    #
    #     total_dices = len(dataset) * 6
    #     correct_rolls_total = total_dices - incorrect_rolls_total
    #     accuracy = correct_rolls_total / total_dices
    #     print(f"Total number of incorrectly identified dice rolls: {incorrect_rolls_total}, Total number of correctly identified dice rolls: {correct_rolls_total}")
    #     print(f"accuracy = {accuracy}")
    #     print(f"Total difference of incorrectly identified rolls: {roll_difference_total}\n")

    def test_kmeans_dataset(self, parameter_configurations, image_dataset='val'):
        dataset = self.set_dataset(image_dataset)

        for config in parameter_configurations:
            print(config)
            # Unpack configuration parameters
            rgb_k, gray_k, median_blur_s, blackhat_open_s, open_s, to_black = config

            # Generate a filename for the results based on the parameters
            file_name = f"{image_dataset}_results_rgbk{rgb_k}_grayk{gray_k}_blur{median_blur_s}_bos{blackhat_open_s}_os{open_s}_{to_black}.txt"
            file_path = os.path.join("./results", file_name)

            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            self.test_one_configuration_kmeans_segmentation(blackhat_open_s, dataset, file_path, gray_k, median_blur_s,
                                                            open_s, rgb_k, to_black)


    def test_one_configuration_kmeans_segmentation(self, blackhat_open_s, dataset, file_path, gray_k, median_blur_s,
                                                   open_s, rgb_k, to_black):
        file = open(file_path, "w")
        file.close()

        with open(file_path, "a") as file:
            count_diff_point_sum = 0
            incorrect_rolls_total = 0
            roll_difference_total = 0

            for img_path in dataset:
                file.write(f'file path = {img_path}')
                # Initialize KmeansSegmentation with current parameters
                kmeans_seg = KmeansSegmentation(
                    img_path=img_path,
                    show=True,
                    rgb_k=rgb_k,
                    gray_k=gray_k,
                    median_blur_s=median_blur_s,
                    blackhat_open_s=blackhat_open_s,
                    open_s=open_s,
                    point_color=to_black,
                )

                extracted_dice_rolls, num_contours = kmeans_seg.find_dice_rolls_with_kmeans()
                rolls, rolls_sum = self.image_loader.extract_dice_rolls_from_filename(img_path)

                count_diff_point_sum += abs(num_contours - rolls_sum)

                incorrect_rolls, roll_difference = self.compare_dice_rolls(extracted_dice_rolls, rolls)
                incorrect_rolls_total += incorrect_rolls
                roll_difference_total += roll_difference

                self.write_one_image_result_summary(extracted_dice_rolls, file, incorrect_rolls, num_contours,
                                                    roll_difference, rolls, rolls_sum)

            self.write_one_test_summary(count_diff_point_sum, dataset, file, incorrect_rolls_total,
                                        roll_difference_total)


    def write_one_image_result_summary(self, extracted_dice_rolls, file, incorrect_rolls, num_contours, roll_difference,
                                       rolls, rolls_sum):
        result = (
            f'number of contours = {num_contours}, correct roll sum = {rolls_sum}\n' +
            f"found rolls: {extracted_dice_rolls}, correct rolls: {rolls}\n"
        )
        #print(result)
        file.write(result)
        details = (
            f"Incorrect dice rolls: {incorrect_rolls}, Difference of rolls: {roll_difference}\n\n"
        )
        #print(details)
        file.write(details)


    def write_one_test_summary(self, count_diff_point_sum, dataset, file, incorrect_rolls_total, roll_difference_total):
        total_dices = len(dataset) * 6
        correct_rolls_total = total_dices - incorrect_rolls_total
        accuracy = correct_rolls_total / total_dices
        summary = (
            f'\nTotal difference between segmented dice points and correct number of points: {count_diff_point_sum}\n'
            f"Total number of incorrectly identified dice rolls: {incorrect_rolls_total}\n"
            f"Total number of correctly identified dice rolls: {correct_rolls_total}\n"
            f"Accuracy = {accuracy}\n"
            f"Total difference of incorrectly identified rolls: {roll_difference_total}\n\n\n"
        )
        print(summary)
        file.write(summary)


    def set_dataset(self, image_dataset):
        if image_dataset == 'val':
            dataset = self.image_loader.val_images
        elif image_dataset == 'test':
            dataset = self.image_loader.test_images
        else:
            dataset = self.image_loader.val_images + self.image_loader.test_images
        return dataset


# parameter_configs = [
#     (3, 7, 41, (30, 30), (10, 10), 'b'),
#     (2, 7, 41, (30, 30), (10, 10), 'b'),
#     (3, 8, 41, (30, 30), (10, 10), 'b'),
#     (2, 8, 41, (30, 30), (10, 10), 'b'),
#     (3, 7, 35, (30, 30), (10, 10), 'b'),
#     (2, 7, 35, (30, 30), (10, 10), 'b'),
#     (3, 8, 35, (30, 30), (10, 10), 'b'),
#     (2, 8, 35, (30, 30), (10, 10), 'b'),
#     (2, 7, 41, (20, 20), (10, 10), 'b'),
#     (3, 7, 35, (20, 20), (10, 10), 'b'), #
#     (3, 8, 35, (20, 20), (10, 10), 'b'),
#     (2, 8, 35, (20, 20), (10, 10), 'b'),
#     (2, 7, 41, (30, 30), (15, 15), 'b'),
#     (3, 7, 35, (30, 30), (15, 15), 'b'), #
#     (3, 8, 35, (30, 30), (15, 15), 'b'),
#     (2, 8, 35, (30, 30), (15, 15), 'b'),
#     (2, 7, 41, (20, 20), (15, 15), 'b'),
#     (3, 7, 35, (20, 20), (15, 15), 'b'), #
#     (3, 8, 35, (20, 20), (15, 15), 'b'),
#     (2, 8, 35, (20, 20), (15, 15), 'b'),
# ]

# test_segmentation = TestKmeansSegmentation()
# test_segmentation.test_kmeans_dataset(parameter_configs, image_dataset='all')
# test_segmentation.test_kmeans_dataset(parameter_configs, image_dataset='test')

parameter_configs = [
    (2, 8, 41, (30, 30), (10, 10), 'b'),
    (2, 7, 41, (30, 30), (10, 10), 'b'),
    (3, 8, 41, (30, 30), (10, 10), 'b'),
    (3, 7, 41, (30, 30), (10, 10), 'b'),

    (2, 7, 39, (30, 30), (10, 10), 'b'),
    (2, 7, 39, (20, 20), (10, 10), 'b'),
    (2, 7, 41, (20, 20), (10, 10), 'b'),
    (2, 7, 39, (20, 20), (15, 15), 'b'),

    (3, 7, 39, (30, 30), (10, 10), 'b'),
    (3, 7, 39, (20, 20), (10, 10), 'b'),
    (3, 7, 41, (20, 20), (10, 10), 'b'),
    (3, 7, 39, (20, 20), (15, 15), 'b'),
]

parameter_configs = [
    (3, 7, 39, (20, 20), (10, 10), 'b'),
]

test_segmentation = TestKmeansSegmentation()
#test_segmentation.test_kmeans_dataset(parameter_configs, image_dataset='test')
test_segmentation.test_kmeans_dataset(parameter_configs, image_dataset='all')
