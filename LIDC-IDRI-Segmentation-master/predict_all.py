import pandas as pd
import argparse
import os
from collections import OrderedDict
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import yaml

from scipy import ndimage as ndi
from scipy.ndimage import label, generate_binary_structure
from tqdm import tqdm

from dataset import MyLidcDataset
from metrics import iou_score,dice_coef,dice_coef2
from utils import AverageMeter, str2bool

from Unet.unet_model import UNet
from UnetNested.Nested_Unet import NestedUNet

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="UNET",
                        help='model name: UNET',choices=['UNET', 'NestedUNET'])
    # Get augmented version?
    parser.add_argument('--augmentation',default=True,type=str2bool,
                help='Shoud we get the augmented version?')

    args = parser.parse_args()

    return args

def save_output(output,output_directory,test_image_paths,counter):
    # This saves the predicted image into a directory. The naming convention will follow PI
    for i in range(output.shape[0]):
        label = test_image_paths[counter][-23:]
        label = label.replace('NI','PD')
        np.save(output_directory+'/'+label,output[i,:,:])
        #print("SAVED",output_directory+label+'.npy')
        counter+=1

    return counter

def main():
    args = vars(parse_args())

    if args['augmentation'] == True:
        NAME = args['name'] + '_with_augmentation'
    else:
        NAME = args['name'] + '_base'

    # load configuration
    with open('model_outputs/{}/config.yml'.format(NAME), 'r') as f:
        config = yaml.full_load(f)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model {}".format(NAME))
    if config['name'] == 'NestedUNET':
        model = NestedUNet(num_classes=1)
    else:
        model = UNet(n_channels=1, n_classes=1, bilinear=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    print("Loading model file from {}".format(NAME))
    model.load_state_dict(torch.load('model_outputs/{}/model.pth'.format(NAME)))
    model = model.cuda()

    # Meta Information
    meta = pd.read_csv(r'D:\Hanun\Dataset\Preprocessing Output\Meta\Fix\cancer_meta.csv')
    # Get train/test label from meta.csv
    meta['original_image'] = meta['original_image'].apply(lambda x: x + '.npy')
    meta['mask_image'] = meta['mask_image'].apply(lambda x: x + '.npy')

    # Get all *npy images into list for Test(True Positive Set)
    test_image_paths = list(meta['original_image'])
    test_mask_paths = list(meta['mask_image'])

    total_patients = len(meta.groupby('patient_id'))

    print("*" * 50)
    print("The lenght of image: {}, mask folders: {} for test".format(len(test_image_paths), len(test_mask_paths)))
    print("Total patient number is :{}".format(total_patients))

    # Directory to save U-Net predict output
    OUTPUT_MASK_DIR = r'D:\Hanun\Dataset\Nodule Output\Predict Segmentation\Segmentation Result'
    print("Saving OUTPUT files in directory {}".format(OUTPUT_MASK_DIR))
    os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)

    test_dataset = MyLidcDataset(test_image_paths, test_mask_paths)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=config['num_workers'])
    model.eval()

    with torch.no_grad():

        counter = 0
        pbar = tqdm(total=len(test_loader))
        for input, target in test_loader:
            input = input.cuda()
            output = model(input)

            output = torch.sigmoid(output)
            output = (output>0.5).float().cpu().numpy()
            output = np.squeeze(output,axis=1)
            #print(output.shape)

            counter = save_output(output,OUTPUT_MASK_DIR,test_image_paths,counter)

            pbar.update(1)
        pbar.close()


if __name__ == '__main__':
    main()