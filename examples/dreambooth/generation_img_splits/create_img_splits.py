import os
import sys

import glob
import random

from tqdm import tqdm
from pprint import pprint

def get_imglist(img_dir, n_images=700, dataset="cityscapes"):
    if dataset == "cityscapes":
        imgs = glob.glob(img_dir + "*/*.png")
    else:
        print("Dataset not supported yet")

    use_imgs = random.sample(imgs, n_images)
    with open(dataset + "_img_split_" + str(n_images) + ".txt", "w") as f:
        for line in use_imgs:
            f.write("%s\n" % line.replace(img_dir, ""))

if __name__ == "__main__":
    IMG_DIR = "/srv/share4/datasets/cityscapesDA/leftImg8bit/train/"
    get_imglist(IMG_DIR)