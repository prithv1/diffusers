import os
import shutil


output_dir = "/coc/flash9/vvijaykumar6/diffusionda/datasets/orig_gta_dataset_n700"
output_img_dir = os.path.join(output_dir, "images")
output_gt_dir = os.path.join(output_dir, "labels")

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

gta_dir = "/srv/share4/datasets/GTA5DA"

suffix_regular = ".png" # used in txt file
suffix = "_labelTrainIds.png"


img_file = "gta_img_split_700.txt"
with open(img_file, 'r') as file:
    img_paths = file.readlines()

img_paths = [line.strip() for line in img_paths]
print(len(img_paths))


for i,img_path in enumerate(img_paths):
    
    in_img_path = os.path.join(gta_dir, "images",img_path)
    in_label_path = os.path.join(gta_dir, "labels",img_path.replace(suffix_regular, suffix))
    in_label_path2 = os.path.join(gta_dir, "labels",img_path)

    

    out_img_path = os.path.join(output_img_dir, img_path)
    out_label_path = os.path.join(output_gt_dir, img_path.replace(suffix_regular, suffix))
    out_label_path2 = os.path.join(output_gt_dir, img_path)
    print(in_img_path, out_img_path)
    # os.symlink(in_img_path, out_img_path)

    print(in_label_path, out_label_path)
    os.symlink(in_label_path2, out_label_path2)








    


