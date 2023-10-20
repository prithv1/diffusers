import os
import shutil


output_dir = "/coc/flash9/vvijaykumar6/diffusionda/datasets/orig_cs_dataset_n700"
output_img_dir = os.path.join(output_dir, "images")
output_gt_dir = os.path.join(output_dir, "labels")

if not os.path.exists(output_img_dir):
    os.makedirs(output_img_dir)

if not os.path.exists(output_gt_dir):
    os.makedirs(output_gt_dir)

gta_dir = "/srv/share4/datasets/cityscapesDA"

suffix_regular = "_leftImg8bit.png" # used in txt file
suffix1 = "_gtFine_labelTrainIds.png"
suffix2 = "_gtFine_polygons.json"


img_file = "cityscapes_img_split_700.txt"
with open(img_file, 'r') as file:
    img_paths = file.readlines()

img_paths = [line.strip() for line in img_paths]
print(len(img_paths))


for i,img_path in enumerate(img_paths):
    
    in_img_path = os.path.join(gta_dir, "leftImg8bit/train",img_path)
    in_label_path = os.path.join(gta_dir, "gtFine/train",img_path.replace(suffix_regular, suffix1))
    in_poly_path = os.path.join(gta_dir, "gtFine/train",img_path.replace(suffix_regular, suffix2))

    

    out_img_path = os.path.join(output_img_dir, img_path)
    out_label_path = os.path.join(output_gt_dir, img_path.replace(suffix_regular, suffix1))
    out_poly_path = os.path.join(output_gt_dir, img_path.replace(suffix_regular, suffix2))
    
    
    place_path = '/'.join(out_img_path.split('/')[:-1])
    # print(place_path)
    if not os.path.exists(place_path):
        os.makedirs(place_path)

    # print(in_img_path, out_img_path)
    os.symlink(in_img_path, out_img_path)


    place_path = '/'.join(out_label_path.split('/')[:-1])
    # print(place_path)
    if not os.path.exists(place_path):
        os.makedirs(place_path)
    
    # print(in_label_path, out_label_path)
    os.symlink(in_label_path, out_label_path)

    # print(in_poly_path, out_poly_path)
    os.symlink(in_poly_path, out_poly_path)








    


