import os
import shutil


output_dir = "/coc/flash9/prithvi/diffusion_DA/hres_fixed_translated_splits/genrun_debug_rml_sd_v15_pp_city_n20_res_512_crop_512_2k_iters_rand1234/new_parse_checkpoint_2k/synthia_city_imglist_n700_use_segmap_1_use_edge_0.5_res1280_gen-seed_10"
output_img_dir = os.path.join(output_dir, "images")
output_gt_dir = os.path.join(output_dir, "labels")

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

synthia_dir = "/coc/flash9/datasets/Synthia/RAND_CITYSCAPES"

suffix_regular = ".png" # used in txt file
suffix = "_labelTrainIds.png"


img_file = "updated_synthia_700_img_split.txt"
with open(img_file, 'r') as file:
    img_paths = file.readlines()

img_paths = [line.strip() for line in img_paths]
print(len(img_paths))


for i,img_path in enumerate(img_paths):
    
    in_img_path = os.path.join(synthia_dir, "RGB",img_path)
    in_label_path = os.path.join(synthia_dir, "GT/LABELS",img_path.replace(suffix_regular, suffix))
    in_label_path2 = os.path.join(synthia_dir, "GT/LABELS",img_path)

    

    # out_img_path = os.path.join(output_img_dir, img_path)
    out_label_path = os.path.join(output_gt_dir, img_path.replace(suffix_regular, suffix))

    # print(out_label_path)
    out_label_path2 = os.path.join(output_gt_dir, img_path)
    # print(in_img_path, out_img_path)
    # os.symlink(in_img_path, out_img_path)

    # print(in_label_path, out_label_path)
    os.symlink(in_label_path, out_label_path)
    print(out_label_path2)
    os.symlink(in_label_path2, out_label_path2)







    


