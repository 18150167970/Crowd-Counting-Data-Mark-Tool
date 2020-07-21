import cv2
import numpy as np
import os
import time
import scipy.io as scio
import shutil


def draw_circle_image(image, point_x, point_y):
    image = cv2.circle(image, (point_x, point_y), 1, (0, 0, 255), 5)
    return image


if __name__ == "__main__":
    GT_paths = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/7_10_data/ground_truth_temp/"
    image_paths = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/7_10_data/images_temp/"
    image_save_paths = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/7_10_data/images/"
    GT_save_paths = "/home/chenli/work/crowd_counting/dataset/ours/mark_dataset/7_10_data/ground_truth/"
    file_names = os.listdir(image_paths)

    total_head_number = 0
    index = 44058
    i = index
    while i < 45918:
        print(i)
        image_path = image_paths + str(i) + '.jpg'
        image_save_path = image_save_paths + str(i) + '.jpg'
        image = cv2.imread(image_path)
        image2 = cv2.resize(image, (1920, 1072))
        points = []

        GT_path = GT_paths + "GT_" + str(i) + ".mat"
        GT_save_path = GT_save_paths + "GT_" + str(i) + ".mat"

        if os.path.exists(GT_path):
            GT_data = scio.loadmat(GT_path)
            gt_annotation_points = GT_data['image_info'] - 1
            for x1, y1 in gt_annotation_points:
                draw_circle_image(image2, x1, y1)

        cv2.imshow("predict", image2)
        k = cv2.waitKey(0)  # 等待并监听键盘活动
        if k == ord('s'):  # 关键点在这一段
            if len(gt_annotation_points) > 0:
                shutil.copy(image_path, image_save_path)
                if os.path.exists(GT_path):
                    shutil.copy(GT_path, GT_save_path)
                print("save image index", i, "success")
        elif k == ord('u'):
            i -= 2
        elif k == ord('q'):
            exit()
        i += 1

    print("total head number", total_head_number)
