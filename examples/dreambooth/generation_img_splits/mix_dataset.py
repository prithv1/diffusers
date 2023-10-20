import os
import shutil
import argparse
import glob
import random
import numpy as np


def get_images(dir, ignore_images=None):
    img_list = glob.glob(dir + "/*.png")

    if len(img_list) == 0:
        img_list = glob.glob(dir + "/**/*.png")
    
    if ignore_images and len(ignore_images) > 0:
        print("no overlapping")
        print(len(img_list))
        img_list = [img for img in img_list if not any(img_id in img for img_id in ignore_images)]
        print(len(img_list))
    return img_list


def symlink_images(image_list, in_img_dir, in_gt_dir, out_img_dir, out_gt_dir):
    img_suffix = ".png"
    gt_suffix = "_labelTrainIds.png"
    count = 0

    img_fail_list = []
    for img_path in image_list:
        img_id = img_path.split('/')[-1]

        in_img_path = img_path
        in_gt_path = os.path.join(in_gt_dir, img_id)
        in_gt_label_path = os.path.join(in_gt_dir, img_id.replace(img_suffix, gt_suffix))

        out_img_path = os.path.join(out_img_dir, img_id)
        out_gt_path = os.path.join(out_gt_dir, img_id)
        out_gt_label_path = os.path.join(out_gt_dir, img_id.replace(img_suffix,gt_suffix))

        if not os.path.exists(in_img_path) or not os.path.exists(in_gt_path) or not os.path.exists(in_gt_label_path):
            img_fail_list.append(img_path)
            continue

        # symlink
        os.symlink(in_img_path, out_img_path)
        os.symlink(in_gt_path, out_gt_path)
        os.symlink(in_gt_label_path, out_gt_label_path)

        # print(in_img_path, out_img_path)
        # print(in_gt_path, out_gt_path)
        # print(in_gt_label_path, out_gt_label_path)
        
    return img_fail_list

parser = argparse.ArgumentParser(description="Perceptual Distance Metrics")
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help="dataset name",
)
parser.add_argument(
    "--root_output_dir",
    type=str,
    default="/coc/flash9/vvijaykumar6/diffusionda/datasets",
    help="root output dir",
)
parser.add_argument(
    "--src_dir",
    type=str,
    default="/coc/flash9/datasets/Synthia/RAND_CITYSCAPES",
    help="src dir",
)
parser.add_argument(
    "--trans_src_dir",
    required=False,
    type=str,
    default="/coc/flash9/prithvi/diffusion_DA/hres_fixed_full_dataset_translation/genrun_debug_rml_sd_v15_pp_city_n20_res_512_crop_512_2k_iters_rand1234/checkpoint_2k/synthia_city_imglist_n24966_use_segmap_1_use_edge_0.5_res1280_gen-seed_10",
    help="trans src dir",
)
parser.add_argument(
    "--img_dir",
    required=False,
    type=str,
    default="RGB",
    help="img dir",
)
parser.add_argument(
    "--gt_dir",
    required=False,
    type=str,
    default="GT/LABELS",
    help="gt dir",
)
parser.add_argument(
    "--num_images",
    required=False,
    type=int,
    default=700,
    help="number of images",
)
parser.add_argument(
    "--mixing_ratio",
    required=False,
    type=float,
    default=0.5,
    help="mixing between dataaset (0) = all source (1) = all trans src dataset",
)
parser.add_argument(
    "--no_overlapping",
    required=False,
    type=bool,
    default=False,
    help="gt dir",
)
args, _ = parser.parse_known_args()

###########################
# DATASET CREATION PARAMS #
###########################
num_images = args.num_images
print(num_images)
mixing_ratio = args.mixing_ratio
no_overlapping = args.no_overlapping

dataset_name = f"synthia_translated_src-cs_n{num_images}_mixing_ratio_{mixing_ratio}_no_overlapping_{no_overlapping}" if args.dataset_name is None else args.dataset_name
output_root_dir = args.root_output_dir
output_dir = os.path.join(output_root_dir, dataset_name)

print(output_dir)

#create output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    img_list = glob.glob(f"{output_dir}/**/*.png", recursive=True)
    if len(img_list) != 0:
        print("Dataset seems to exist already")
        exit(0)




# create "images" and "labels" dirs
output_img_dir = os.path.join(output_dir, "images")
output_gt_dir = os.path.join(output_dir, "labels")
if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)
if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)



# src info
src_dir = args.src_dir
src_img_dir = os.path.join(src_dir, args.img_dir)
src_gt_dir = os.path.join(src_dir, args.gt_dir)

# trans src info 
trans_src_dir = args.trans_src_dir
trans_src_img_dir =  os.path.join(trans_src_dir, "images")
trans_src_gt_dir =  os.path.join(trans_src_dir, "labels")


# make img list
src_mixing_num = int(num_images * (1-mixing_ratio))
trans_src_mixing_num = int(num_images - (src_mixing_num))


# get trans src list
trans_src_img_list = get_images(trans_src_img_dir)

trans_src_list = []
src_list = []
if no_overlapping:
    trans_src_list = random.sample(trans_src_img_list, trans_src_mixing_num)
    img_ids = [img_id.split('/')[-1] for img_id in trans_src_list]

    src_img_list = get_images(src_img_dir, ignore_images=img_ids)
    src_list = random.sample(src_img_list, src_mixing_num)




else:
    trans_src_list = random.sample(trans_src_img_list, trans_src_mixing_num)
    src_img_list = get_images(src_img_dir)
    src_list = random.sample(src_img_list, src_mixing_num)



trans_src_img_fail_list = symlink_images(trans_src_list, trans_src_img_dir, trans_src_gt_dir, output_img_dir, output_gt_dir)
src_img_fail_list = symlink_images(src_list, src_img_dir, src_gt_dir, output_img_dir, output_gt_dir)


final_img_list = trans_src_list + src_list

output_file = dataset_name + ".txt"
with open(output_file, 'w') as file:
    for img in final_img_list:
        file.write(img + "\n")

print("*****************")
print("failed img list from translated src", trans_src_img_fail_list)
print("failed img list from src", src_img_fail_list)









    


