import os
import shutil


output_dir = "/coc/flash9/vvijaykumar6/diffusionda/datasets/cs_acdc_combined_700_seed_1"
output_img_dir = os.path.join(output_dir, "images")
output_gt_dir = os.path.join(output_dir, "labels")

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

#Cityscapes dir (grab all gt from og cityscapes dir)
cs_label_dir = "/srv/share4/datasets/cityscapesDA/gtFine/train"
#grab all gt from og cityscapes dir

suffix_regular = "leftImg8bit.png" # used in txt file
#replace with these:
suffix = ["gtFine_polygons.json", "gtFine_labelTrainIds.png", "gtFine_color.png"]

#cs--> acdc translated image dir
fog_dir = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_fog_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
night_dir = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_night_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
rain_dir = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_rain_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
snow_dir = "/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_snow_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"


# read split file for cityscape images
img_file = "cityscapes_img_split_700.txt"
with open(img_file, 'r') as file:
    img_paths = file.readlines()

img_paths = [line.strip() for line in img_paths]
print(len(img_paths))

#partitioning
img_paths_len = len(img_paths)
partition = img_paths_len // 4


# to iterate through all condition dirs
index = 0
acdc_conditions = [fog_dir, night_dir, rain_dir, snow_dir]
translated_base_dir = acdc_conditions[index]

for i,img_path in enumerate(img_paths):
    if i != 0 and i % partition == 0:
        index += 1
        translated_base_dir = acdc_conditions[index]
        print(translated_base_dir)
    
    place = img_path.split('/')[0]
    img_id =  img_path.split('/')[-1]

    full_img_path = os.path.join(translated_base_dir, img_path)


    # translated image
    dest_dir_image = os.path.join(output_img_dir,place)
    if not os.path.exists(dest_dir_image):
        os.makedirs(dest_dir_image)
    shutil.copy(full_img_path, dest_dir_image)


    #ground truths
    dest_dir_label = os.path.join(output_gt_dir,place)
    if not os.path.exists(dest_dir_label):
        os.makedirs(dest_dir_label)

    for x in suffix:
        gt_file = os.path.join(cs_label_dir,img_path.replace(suffix_regular, x))
        shutil.copy(gt_file, dest_dir_label)









    


