import numpy as np
import cv2
import os

picture_folder = r'E:\Koding\Python\Thesis\Object Detection\Object Picture'
size_folder = r'E:\Koding\Python\Thesis\Object Detection\Predict Size'
nodule_folder = r'E:\Koding\Python\Thesis\Object Detection\PNG Image'
new_folder = r'E:\Koding\Python\Thesis\Object Detection\Combine Picture'
annotation_folder = r'E:\Koding\Python\Thesis\Object Detection\Annotation Picture'
dataset_folder = r'E:\Koding\Python\Thesis\Object Detection\Dataset'

for folder_name in os.listdir(picture_folder):
    print(folder_name)
    folder_img = os.path.join(picture_folder, folder_name)

    for file_name in os.listdir(folder_img):
        if "CN" in file_name:
            new_file_name = file_name.replace("CN", "PD")
        elif "NI" in file_name:
            new_file_name = file_name.replace("NI", "PD")
        else:
            print("Not Found!!!")

        picture_path = os.path.join(folder_img, file_name)
        size_path = os.path.join(size_folder, new_file_name)
        nodule_path = os.path.join(nodule_folder, new_file_name)
        annotation_path = os.path.join(annotation_folder, new_file_name)

        if(os.path.exists(picture_path) and os.path.exists(size_path) and os.path.exists(nodule_path)):
            picture = cv2.imread(picture_path)
            picture_size = cv2.imread(size_path)
            picture_nodule = cv2.imread(nodule_path)

            file_save = os.path.join(dataset_folder, new_file_name)
            cv2.imwrite(file_save, picture)

            dst = cv2.addWeighted(picture, 0.7, picture_nodule, 0.7, 0)
            file_save = os.path.join(annotation_folder, new_file_name)
            cv2.imwrite(file_save, dst)

            img_arr = np.hstack((picture, dst, picture_size))
            file_save = os.path.join(new_folder, new_file_name)
            cv2.imwrite(file_save, img_arr)





