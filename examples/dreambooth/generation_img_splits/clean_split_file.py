import os
import json


# file_name = 'gta_img_split_2.txt'
file_name = 'synthia_img_split_700.txt'
with open(file_name, 'r') as file:
    img_list = file.readlines()


img_list_edited = [img.split('/')[-1] for img in img_list]


with open(file_name, 'w') as file:
    file.writelines(img_list_edited)