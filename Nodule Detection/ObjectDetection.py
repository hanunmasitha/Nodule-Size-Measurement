import numpy as np
import cv2
import os

PREDICT_DIR = r'E:\Koding\Python\Thesis\Preprocessing Output\Image'
SAVE_DIR = r'E:\Koding\Python\Thesis\Object Detection\Object Picture'
mode = 0o666

for folder_name in os.listdir(PREDICT_DIR):
    folder_save = os.path.join(SAVE_DIR, folder_name)
    folder_img = os.path.join(PREDICT_DIR, folder_name)

    try:
        os.makedirs(folder_save, exist_ok=True)
        print("Directory '%s' created successfully" % folder_name)
    except OSError as error:
        print("Directory '%s' can not be created" % folder_name)

    for file_name in os.listdir(folder_img):
        file_path = os.path.join(folder_img, file_name)

        image = np.load(file_path)
        # normalize file from 0-1 to 0-255
        norm_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        norm_image = norm_image.astype(np.uint8)

        pre, ext = os.path.splitext(file_name)
        new_file_name = pre + '.png'
        file_save = os.path.join(folder_save, new_file_name)
        cv2.imwrite(file_save, norm_image)


