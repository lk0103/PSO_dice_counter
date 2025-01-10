import os

from Image_loader import ImageLoader
from Blackhat_proximity import BlackHat as BHP
from BlackHat_Cluster import BlackHat as BHC
from Canny import Canny

class TestAll():
    def __init__(self):
        self.image_loader = ImageLoader()

    def recalculate(self,result):
        new_result = []
        for n, side in enumerate(result):
            for j in range(side):
                new_result.append(n+1)
        return new_result

    def compare_dice_rolls(self, found_rolls, correct_rolls):

        # Measure 1: Count of incorrectly identified dice rolls
        incorrect_rolls = sum(1 for f, c in zip(found_rolls, correct_rolls) if f != c)

        # Measure 2: Total difference in dice rolls
        roll_difference = sum(abs(f - c) for f, c in zip(found_rolls, correct_rolls))

        return incorrect_rolls, roll_difference

    def test_BlackHat_Cluster_dataset(self, image_dataset='val'):
        dataset = self.set_dataset(image_dataset)
        file_name = f"BlackHat_Cluster.txt"
        file_path = os.path.join("./results", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open("filename", "w").close()

        with open(file_path, "a") as file:
            count_diff_point_sum = 0
            incorrect_rolls_total = 0
            roll_difference_total = 0

            for img_path in dataset:
                file.write(f'file path = {img_path}')
                bhc = BHC(img_path)

                extracted_dice_rolls = bhc.test()


                extracted_dice_rolls = self.recalculate(extracted_dice_rolls)
                rolls, rolls_sum = self.image_loader.extract_dice_rolls_from_filename(img_path)

                count_diff_point_sum += 0  # ' idk ze co' #abs(num_contours - rolls_sum)

                incorrect_rolls, roll_difference = self.compare_dice_rolls(extracted_dice_rolls, rolls)
                incorrect_rolls_total += incorrect_rolls
                roll_difference_total += roll_difference

                self.write_one_image_result_summary(extracted_dice_rolls, file, incorrect_rolls,
                                                    roll_difference, rolls, rolls_sum)

            self.write_one_test_summary(count_diff_point_sum, dataset, file, incorrect_rolls_total,
                                        roll_difference_total)

    def test_BlackHat_Proximity_dataset(self, image_dataset='val'):
        dataset = self.set_dataset(image_dataset)
        file_name = f"BlackHat_Proximity.txt"
        file_path = os.path.join("./results", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open("filename", "w").close()

        with open(file_path, "a") as file:
            count_diff_point_sum = 0
            incorrect_rolls_total = 0
            roll_difference_total = 0

            for img_path in dataset:
                file.write(f'file path = {img_path}')
                bhp = BHP(img_path)

                extracted_dice_rolls = bhp.test()

                extracted_dice_rolls = self.recalculate(extracted_dice_rolls)
                rolls, rolls_sum = self.image_loader.extract_dice_rolls_from_filename(img_path)

                count_diff_point_sum += 0 #' idk ze co' #abs(num_contours - rolls_sum)

                incorrect_rolls, roll_difference = self.compare_dice_rolls(extracted_dice_rolls, rolls)
                incorrect_rolls_total += incorrect_rolls
                roll_difference_total += roll_difference

                self.write_one_image_result_summary(extracted_dice_rolls, file, incorrect_rolls,
                                                    roll_difference, rolls, rolls_sum)

            self.write_one_test_summary(count_diff_point_sum, dataset, file, incorrect_rolls_total,
                                        roll_difference_total)

    def write_one_image_result_summary(self, extracted_dice_rolls, file, incorrect_rolls, roll_difference,
                                       rolls, rolls_sum):
        result = (
                f'correct roll sum = {rolls_sum}\n' +
                f"found rolls: {extracted_dice_rolls}, correct rolls: {rolls}\n"
        )
        # print(result)
        file.write(result)
        details = (
            f"Incorrect dice rolls: {incorrect_rolls}, Difference of rolls: {roll_difference}\n\n"
        )
        # print(details)
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

    def test_Canny_Proximity_dataset(self, image_dataset='val'):
        dataset = self.set_dataset(image_dataset)
        file_name = f"Canny_Proximity.txt"
        file_path = os.path.join("./results", file_name)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        open("filename", "w").close()

        with open(file_path, "a") as file:
            count_diff_point_sum = 0
            incorrect_rolls_total = 0
            roll_difference_total = 0

            for img_path in dataset:
                file.write(f'file path = {img_path}')
                canny = Canny(img_path)

                extracted_dice_rolls = canny.test()

                extracted_dice_rolls = self.recalculate(extracted_dice_rolls)
                rolls, rolls_sum = self.image_loader.extract_dice_rolls_from_filename(img_path)

                count_diff_point_sum += 0  # ' idk ze co' #abs(num_contours - rolls_sum)

                incorrect_rolls, roll_difference = self.compare_dice_rolls(extracted_dice_rolls, rolls)
                incorrect_rolls_total += incorrect_rolls
                roll_difference_total += roll_difference

                self.write_one_image_result_summary(extracted_dice_rolls, file, incorrect_rolls,
                                                    roll_difference, rolls, rolls_sum)

            self.write_one_test_summary(count_diff_point_sum, dataset, file, incorrect_rolls_total,
                                        roll_difference_total)



    def test_Template_Match_dataset(self, image_dataset='val'):
        dataset = self.set_dataset(image_dataset)

    def set_dataset(self, image_dataset):
        if image_dataset == 'val':
            dataset = self.image_loader.val_images
        elif image_dataset == 'test':
            dataset = self.image_loader.test_images
        else:
            dataset = self.image_loader.val_images + self.image_loader.test_images
        return dataset





test_BH_prox = TestAll()
#test_BH_prox.test_BlackHat_Cluster_dataset(image_dataset='all')
test_BH_prox.test_Canny_Proximity_dataset(image_dataset='test')
test_BH_prox.test_Canny_Proximity_dataset(image_dataset='all')
#test_BH_prox.test_BlackHat_Proximity_dataset( image_dataset='all')
# test_BH_prox.test_BlackHat_Proximity_dataset( image_dataset='all')