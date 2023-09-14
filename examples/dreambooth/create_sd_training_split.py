import os
import sys
import glob
import random

import numpy as np

from tqdm import tqdm
from pprint import pprint

random.seed(1234)

def get_imglist(directory, dset="cityscapes"):
    imlist = None
    if dset == "cityscapes":
        imlist = glob.glob(directory + "/*/*.png")
    elif dset == "mapillary":
        imlist = glob.glob(directory + "/*.jpg")
    elif dset == "nuscenes":
        imlist = glob.glob(directory + "/*.jpg")
    else:
        print("Dataset not supported")
    return imlist

def create_sd_dataset(directory, n_images, destination, dset="cityscapes"):
    imlist = get_imglist(directory, dset)
    # print(imlist)
    print(len(imlist))
    imgs = random.sample(imlist, n_images)
    
    dst_dir = os.path.join("sd_datasets", destination)
    if not os.path.exists(dst_dir):
        os.mkdir(dst_dir)
        
    print("Creating Stable Diffusion fine-tuning dataset..")
    for i in tqdm(range(len(imgs))):
        img = imgs[i]
        img_src = img
        img_dst = os.path.join(dst_dir, img.split("/")[-1])
        os.symlink(img_src, img_dst)
    
if __name__ == "__main__":
    DIRECTORY = "/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
    N_IMAGES = 500
    DESTINATION = "sd_v3_dataset_city_n500_rand1234"
    create_sd_dataset(
        DIRECTORY,
        N_IMAGES,
        DESTINATION,
    )
    
    # DIRECTORY = "/srv/share4/datasets/mapillary/training/images"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_mapillary_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "mapillary",
    # )
    
    # DIRECTORY = "/srv/datasets/nuScenes-v1.0/nuScenes-mini/samples/CAM_FRONT"
    # N_IMAGES = 400
    # DESTINATION = "sd_v3_dataset_nuscenes_n400_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "nuscenes",
    # )
        
    