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
    elif dset == "nyuv2":
        imlist = glob.glob(directory + "/*.png")
    elif dset == "gta_anno":
        imlist = glob.glob(directory + "/*.png")
    else:
        print("Dataset not supported")
    return imlist

def create_sd_dataset(directory, n_images, destination, dset="cityscapes", filter_key=None):
    imlist = get_imglist(directory, dset)
    
    if filter_key is not None:
        imlist = [x for x in imlist if "skybox" in x]
        imlist = [x for x in imlist if "sami" in x]
    
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
    # DIRECTORY = "/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_city_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    # )
    
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
    
    # DIRECTORY = "/srv/share4/datasets/NYUv2/data/image/train"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_nyuv2_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "nyuv2",
    # )
    
    # # Train a diffusion model on GTAV driving scene segmentation maps
    # DIRECTORY = "/srv/datasets/GTA5/labels"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_gtav_semseg_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "gta_anno",
    # )
    
    
    # Train a diffusion model on SYNTHIA driving scene segmentation maps
    # DIRECTORY = "/srv/datasets/SYNTHIA/RAND_CITYSCAPES/RGB"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_synthia_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "gta_anno",
    # )
    
    # /srv/datasets/bdd100k/bdd100k/images/10k/train
    # # Train a diffusion model on SYNTHIA driving scene segmentation maps
    # DIRECTORY = "/srv/datasets/bdd100k/bdd100k/images/10k/train"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_bdd10k_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "nuscenes",
    # )
    
    # # Train a diffusion model on SYNTHIA driving scene segmentation maps
    # DIRECTORY = "/srv/datasets/matterport/v1/unzipped_scans/yZVvKaJZghh_theta_2020_07_23/matterport_skybox_images"
    # N_IMAGES = 350
    # DESTINATION = "sd_v3_dataset_coda_n350_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "nuscenes",
    #     "coda",
    # )
    
    # Train a diffusion model on GTAV driving scene segmentation maps
    # DIRECTORY = "/srv/share4/datasets/GTA5DA/images"
    # N_IMAGES = 500
    # DESTINATION = "sd_v3_dataset_gtav_n500_rand1234"
    # create_sd_dataset(
    #     DIRECTORY,
    #     N_IMAGES,
    #     DESTINATION,
    #     "gta_anno",
    # )
    
    # HM3D as Real Data
    DIRECTORY = "/srv/share4/datasets/HM3D_Sim/hm3d_sim"
    N_IMAGES = 500
    DESTINATION = "sd_v3_dataset_hm3d_n500_rand1234"
    create_sd_dataset(
        DIRECTORY,
        N_IMAGES,
        DESTINATION,
        "nuscenes",
    )