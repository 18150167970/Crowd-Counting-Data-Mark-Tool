# encoding=utf-8
import cv2 as cv
import os
import numpy as np
import scipy.io as scio
import json


config_path = 'dot_mark_config.json'
CONFIG = json.load(open(config_path, 'r', encoding='UTF-8'))
dot_thickness = CONFIG["dot_thickness"]
class IMG:
    data = None
    image = None
    points = []


def draw_point_image(image, points):
    for point_x, point_y in points:
        draw_circle_image(image, point_x, point_y)
    return image


def draw_circle_image(image, point_x, point_y):
    image = cv.circle(image, (point_x, point_y), 1, (0, 0, 255), dot_thickness)
    return image


def mouse_event(event, x, y, flags, item):
    if event == cv.EVENT_MBUTTONDOWN:
        mins = 1000000
        mins_x = -1
        mins_y = -1
        mins_index = -1

        for index, (x2, y2) in enumerate(item.points):
            if mins > abs(x2 - x) + abs(y2 - y):
                mins = abs(x2 - x) + abs(y2 - y)
                mins_x = x2
                mins_y = y2
                mins_index = index
        if mins_index >= 0 and abs(mins_x - x) + abs(mins_y - y) <= 6:
            del item.points[mins_index]

        item.data = item.image.copy()
        item.data = draw_point_image(item.data, item.points)
        cv.imshow("image", item.data)
    elif event == cv.EVENT_LBUTTONDOWN:
        cv.circle(item.data, (x, y), 1, (0, 0, 255), dot_thickness)
        item.points.append((x, y))
        cv.imshow("image", item.data)


def item_init(item, img):
    item.points = []
    item.image = img.copy()
    item.data = img.copy()
    return item


def ground_truth_load(item, GT_save_path):
    GT_data = scio.loadmat(GT_save_path)
    if GT_data['image_info'].dtype == 'uint16':
        gt_annotation_points = GT_data['image_info'] - 1
        item.points = list(gt_annotation_points)
        item.image = img.copy()
        item.data = img.copy()
        item.data = draw_point_image(item.data, item.points)
    else:
        item = item_init(item, img)
    return item


if __name__ == '__main__':
    image_path = CONFIG["image_path"]
    save_path = CONFIG["save_path"]
    image_star_index = CONFIG["image_star_index"]

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_names = os.listdir(image_path)
    img_names = sorted(img_names)
    cv.namedWindow("image")
    image_index = image_star_index
    CONFIG["total_image_number"] = len(img_names)
    while image_index < len(img_names):
        print("current image index is", image_index)
        img_name = img_names[image_index]
        img = cv.imread(image_path + img_name)
        item = IMG()
        GT_save_path = save_path + "GT_" + img_name[:-4] + ".mat"
        if os.path.exists(GT_save_path):
            item = ground_truth_load(item, GT_save_path)
        else:
            item = item_init(item, img)

        while 1:
            cv.setMouseCallback("image", mouse_event, item)
            cv.imshow("image", item.data)
            k = cv.waitKey(0)
            if k == ord('s'):
                if len(item.points):
                    GT_data = {'image_info': np.array(item.points, np.uint16) + 1}
                    scio.savemat(GT_save_path, GT_data)
                    print("save image index", image_index, "success")
                else:
                    if os.path.exists(GT_save_path):
                        os.remove(GT_save_path)
                break
            if k == ord('n'):
                break
            if k == ord('b'):
                item.data = item.image.copy()
                if len(item.points):
                    item.points.pop()
                    item.data = draw_point_image(item.data, item.points)
            if k == ord('u'):
                image_index -= 2
                break
            if k == ord('c'):
                item = item_init(item, img)
            if k == ord('q'):
                exit()
        image_index += 1
        CONFIG['image_star_index'] = image_index
        json.dump(CONFIG, open(config_path, 'w', encoding='UTF-8'))
