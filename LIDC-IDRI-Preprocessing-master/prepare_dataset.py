import sys
import os
from pathlib import Path
import glob
from configparser import ConfigParser
import pandas as pd
import numpy as np
import warnings
import pylidc as pl
from tqdm import tqdm
from statistics import median_high
from sklearn.model_selection import train_test_split

from utils import is_dir_path,segment_lung
from pylidc.utils import consensus
from PIL import Image

warnings.filterwarnings(action='ignore')

# Read the configuration file generated from config_file_create.py
parser = ConfigParser()
parser.read('lung.conf')

#Get Directory setting
DICOM_DIR = is_dir_path(parser.get('prepare_dataset','LIDC_DICOM_PATH'))
MASK_DIR = is_dir_path(parser.get('prepare_dataset','MASK_PATH'))
IMAGE_DIR = is_dir_path(parser.get('prepare_dataset','IMAGE_PATH'))
CLEAN_DIR_IMAGE = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_IMAGE'))
CLEAN_DIR_MASK = is_dir_path(parser.get('prepare_dataset','CLEAN_PATH_MASK'))
META_DIR = is_dir_path(parser.get('prepare_dataset','META_PATH'))

#Hyper Parameter setting for prepare dataset function
mask_threshold = parser.getint('prepare_dataset','Mask_Threshold')

#Hyper Parameter setting for pylidc
confidence_level = parser.getfloat('pylidc','confidence_level')
padding = parser.getint('pylidc','padding_size')

class MakeDataSet:
    def __init__(self, LIDC_Patients_list, IMAGE_DIR, MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR, mask_threshold, padding, confidence_level=0.5):
        self.IDRI_list = LIDC_Patients_list
        self.img_path = IMAGE_DIR
        self.mask_path = MASK_DIR
        self.clean_path_img = CLEAN_DIR_IMAGE
        self.clean_path_mask = CLEAN_DIR_MASK
        self.meta_path = META_DIR
        self.mask_threshold = mask_threshold
        self.c_level = confidence_level
        self.padding = [(padding,padding),(padding,padding),(0,0)]
        self.meta = pd.DataFrame(index=[],columns=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])


    def calculate_malignancy(self,nodule):
        # Calculate the malignancy of a nodule with the annotations made by 4 doctors. Return median high of the annotated cancer, True or False label for cancer
        # if median high is above 3, we return a label True for cancer
        # if it is below 3, we return a label False for non-cancer
        # if it is 3, we return ambiguous
        list_of_malignancy =[]
        for annotation in nodule:
            list_of_malignancy.append(annotation.malignancy)

        malignancy = median_high(list_of_malignancy)
        if  malignancy > 3:
            return malignancy,True
        elif malignancy < 3:
            return malignancy, False
        else:
            return malignancy, 'Ambiguous'
    def save_meta(self,meta_list):
        """Saves the information of nodule to csv file"""
        tmp = pd.Series(meta_list,index=['patient_id','nodule_no','slice_no','original_image','mask_image','malignancy','is_cancer','is_clean'])
        self.meta = self.meta.append(tmp,ignore_index=True)

    def prepare_dataset(self):
        # This is to name each image and mask
        prefix = [str(x).zfill(3) for x in range(1000)]

        # Make directory
        if not os.path.exists(self.img_path):
            os.makedirs(self.img_path)
        if not os.path.exists(self.mask_path):
            os.makedirs(self.mask_path)
        if not os.path.exists(self.clean_path_img):
            os.makedirs(self.clean_path_img)
        if not os.path.exists(self.clean_path_mask):
            os.makedirs(self.clean_path_mask)
        if not os.path.exists(self.meta_path):
            os.makedirs(self.meta_path)

        IMAGE_DIR = Path(self.img_path)
        MASK_DIR = Path(self.mask_path)
        CLEAN_DIR_IMAGE = Path(self.clean_path_img)
        CLEAN_DIR_MASK = Path(self.clean_path_mask)

        for patient in tqdm(self.IDRI_list):
            pid = patient #LIDC-IDRI-0001~
            scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
            nodules_annotation = scan.cluster_annotations()
            vol = scan.to_volume()
            print("Patient ID: {} Dicom Shape: {} Number of Annotated Nodules: {}".format(pid,vol.shape,len(nodules_annotation)))

            patient_image_dir = IMAGE_DIR / pid
            patient_mask_dir = MASK_DIR / pid
            Path(patient_image_dir).mkdir(parents=True, exist_ok=True)
            Path(patient_mask_dir).mkdir(parents=True, exist_ok=True)

            if len(nodules_annotation) > 0:
                # Patients with nodules
                for nodule_idx, nodule in enumerate(nodules_annotation):
                # Call nodule images. Each Patient will have at maximum 4 annotations as there are only 4 doctors
                # This current for loop iterates over total number of nodules in a single patient
                    mask, cbbox, masks = consensus(nodule,self.c_level,self.padding)
                    lung_np_array = vol[cbbox]

                    # We calculate the malignancy information
                    malignancy, cancer_label = self.calculate_malignancy(nodule)

                    for nodule_slice in range(mask.shape[2]):
                        # This second for loop iterates over each single nodule.
                        # There are some mask sizes that are too small. These may hinder training.
                        if np.sum(mask[:,:,nodule_slice]) <= self.mask_threshold:
                            continue
                        # Segment Lung part only
                        lung_segmented_np_array = lung_np_array[:,:,nodule_slice]
                        # I am not sure why but some values are stored as -0. <- this may result in datatype error in pytorch training # Not sure
                        lung_segmented_np_array[lung_segmented_np_array==-0] =0
                        # This itereates through the slices of a single nodule
                        # Naming of each file: NI= Nodule Image, MA= Mask Original
                        nodule_name = "{}_NI{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])
                        mask_name = "{}_MA{}_slice{}".format(pid[-4:],prefix[nodule_idx],prefix[nodule_slice])

                        nodule_name = patient_image_dir / nodule_name
                        mask_name = patient_mask_dir / mask_name

                        meta_list = [pid[-4:],nodule_idx,prefix[nodule_slice],nodule_name,mask_name,malignancy,cancer_label,False]

                        #print(meta_list)
                        self.save_meta(meta_list)
                        np.save(nodule_name,lung_segmented_np_array)
                        np.save(mask_name,mask[:,:,nodule_slice])
            else:
                print("Clean Dataset",pid)
                patient_clean_dir_image = CLEAN_DIR_IMAGE / pid
                patient_clean_dir_mask = CLEAN_DIR_MASK / pid
                Path(patient_clean_dir_image).mkdir(parents=True, exist_ok=True)
                Path(patient_clean_dir_mask).mkdir(parents=True, exist_ok=True)
                #There are patients that don't have nodule at all. Meaning, its a clean dataset. We need to use this for validation
                for slice in range(vol.shape[2]):
                    if slice >50:
                        break
                    lung_segmented_np_array = vol[:,:,slice]
                    lung_segmented_np_array[lung_segmented_np_array==-0] =0
                    lung_mask = np.zeros_like(lung_segmented_np_array)

                    #CN= CleanNodule, CM = CleanMask
                    nodule_name = "{}_CN001_slice{}".format(pid[-4:],prefix[slice])
                    mask_name = "{}_CM001_slice{}".format(pid[-4:],prefix[slice])

                    nodule_name = patient_clean_dir_image / nodule_name
                    mask_name = patient_clean_dir_mask / mask_name

                    meta_list = [pid[-4:],slice,prefix[slice],nodule_name,mask_name,0,False,True]
                    self.save_meta(meta_list)
                    np.save(nodule_name, lung_segmented_np_array)
                    np.save(mask_name, lung_mask)



        print("Saved Meta data")
        self.meta.to_csv(self.meta_path+'meta_info.csv',index=False)

    def is_nodule(row):
        if row[5:7] == 'NI':
            return True
        else:
            return False

    def is_train(row, train, val, test):
        if row in train:
            return 'Train'
        elif row in val:
            return 'Validation'
        else:
            return 'Test'

    def create_label_segmentation(meta):
        patient_id = list(np.unique(meta['patient_id']))
        train_patient, test_patient = train_test_split(patient_id, test_size=0.2)
        train_patient, val_patient = train_test_split(train_patient, test_size=0.25)
        print(len(train_patient), len(val_patient), len(test_patient))

        meta['data_split'] = meta['patient_id'].apply(
            lambda row: MakeDataSet.is_train(row, train_patient, val_patient, test_patient))

        return meta

    def make_label(self):
        label_meta = pd.read_csv(r'D:\Hanun\Dataset\Preprocessing Output\Meta\meta_info.csv')
        label_meta['is_nodule'] = False
        for index, row in label_meta.iterrows():
            name = row['original_image'].split('\\')
            label_meta['is_nodule'][index] = MakeDataSet.is_nodule(name[-1])

        # Lets separate Clean meta and meta data
        clean_meta = label_meta[label_meta['is_nodule'] == False]
        clean_meta.reset_index(inplace=True)
        cancer_meta = label_meta[label_meta['is_nodule'] == True]
        cancer_meta.reset_index(inplace=True)

        # We need to train/test split independently for clean_meta, meta
        cancer_meta = MakeDataSet.create_label_segmentation(cancer_meta)
        clean_meta = MakeDataSet.create_label_segmentation(clean_meta)

        # Clean Meta only stores meta information of patients without nodules.
        cancer_meta.to_csv(r'D:\Hanun\Dataset\Preprocessing Output\Meta\Fix\cancer_meta.csv')
        clean_meta.to_csv(r'D:\Hanun\Dataset\Preprocessing Output\Meta\Fix\clean_meta.csv')




if __name__ == '__main__':
    # I found out that simply using os.listdir() includes the gitignore file 
    LIDC_IDRI_list= [f for f in os.listdir(DICOM_DIR) if not f.startswith('.')]
    LIDC_IDRI_list.sort()


    test= MakeDataSet(LIDC_IDRI_list,IMAGE_DIR,MASK_DIR,CLEAN_DIR_IMAGE,CLEAN_DIR_MASK,META_DIR,mask_threshold,padding,confidence_level)
    test.prepare_dataset()
    test.make_label()
