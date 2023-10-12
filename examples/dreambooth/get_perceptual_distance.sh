
### Sim2Real (GTAV --> Cityscapes)###

# EXP_NAME="GTA_CS_Metrics_FID"
# # TGT_DIR="/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/sd_datasets/sd_v3_dataset_city_n500_rand1234"
# TGT_DIR="/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
# TGT_LIST="./generation_img_splits/sd_v3_dataset_city_n500_rand1234.txt"
# TRANS_DIR="/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_city_n500_rand1234/checkpoint_2k/gta_cs_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
# # SRC_DIR="/nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/generation_img_splits/source_dirs/gtav_gen_source"
# SRC_DIR="/srv/share4/datasets/GTA5DA/images"
# SRC_LIST="./generation_img_splits/gta_img_split_700.txt"
# SRC_LBL_DIR="/srv/share4/datasets/GTA5DA/labels"
# SRC_DATASET="gtav"

# python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1052 1914 --src_dataset $SRC_DATASET  |& tee -a "metrics/$EXP_NAME.txt"



## Sim2Real (GTAV --> MAP) ###
EXP_NAME="GTA_MAP_Metrics_FID"
TGT_DIR="/srv/share4/datasets/mapillary/training/images"
TGT_LIST="./generation_img_splits/sd_v3_dataset_mapillary_n500_rand1234.txt"
TRANS_DIR="/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_map_n500_rand1234/checkpoint_2k/gta_map_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/GTA5DA/images"
SRC_LIST="./generation_img_splits/gta_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/GTA5DA/labels"
SRC_DATASET="mapillary"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1052 1914 --src_dataset $SRC_DATASET |& tee "metrics/$EXP_NAME.txt"



# ### Sim2Real (GTAV --> BDD10k) ###

EXP_NAME="GTA_BDD_Metrics_FID"
TGT_DIR="/srv/datasets/bdd100k/bdd100k/images/10k/train"
TGT_LIST="./generation_img_splits/sd_v3_dataset_bdd10k_n500_rand1234.txt"
TRANS_DIR="/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_bdd10k_n500_rand1234/checkpoint_2k/gta_bdd100k_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/GTA5DA/images"
SRC_LIST="./generation_img_splits/gta_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/GTA5DA/labels"
SRC_DATASET="bdd10k"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1052 1914 --src_dataset $SRC_DATASET |& tee -a "metrics/$EXP_NAME.txt"


### Sim2Real (GTAV --> SYNTHIA) ###

EXP_NAME="GTA_SYNTHIA_Metrics_FID"
TGT_DIR="/srv/datasets/SYNTHIA/RAND_CITYSCAPES/RGB/"
TGT_LIST="./generation_img_splits/sd_v3_dataset_synthia_n500_rand1234.txt"
TRANS_DIR="/srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_synthia_n500_rand1234/checkpoint_2k/gta_synthia_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/GTA5DA/images"
SRC_LIST="./generation_img_splits/gta_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/GTA5DA/labels"
SRC_DATASET="synthia"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1052 1914 --src_dataset $SRC_DATASET |& tee -a "metrics/$EXP_NAME.txt"

### Cityscapes --> ACDC FOG ###

EXP_NAME="CS_ACDC_FOG_Metrics_FID"
TGT_DIR="/srv/share4/datasets/ACDC/rgb_anon/fog/train"
TGT_LIST="./generation_img_splits/sd_v3_dataset_acdc_tr_fog_n400_rand1234.txt"
TRANS_DIR="/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_fog_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
SRC_LIST="./generation_img_splits/cityscapes_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/cityscapesDA/gtFine/train"
SRC_DATASET="cityscapes"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1024 2048 --src_dataset $SRC_DATASET  |& tee -a "metrics/$EXP_NAME.txt"


### Cityscapes --> ACDC RAIN ###

EXP_NAME="CS_ACDC_RAIN_Metrics_FID"
TGT_DIR="/srv/share4/datasets/ACDC/rgb_anon/rain/train"
TGT_LIST="./generation_img_splits/sd_v3_dataset_acdc_tr_rain_n400_rand1234.txt"
TRANS_DIR="/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_rain_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
SRC_LIST="./generation_img_splits/cityscapes_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/cityscapesDA/gtFine/train"
SRC_DATASET="cityscapes"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1024 2048 --src_dataset $SRC_DATASET  |& tee -a "metrics/$EXP_NAME.txt"

### Cityscapes --> ACDC SNOW ###

EXP_NAME="CS_ACDC_SNOW_Metrics_FID"
TGT_DIR="/srv/share4/datasets/ACDC/rgb_anon/snow/train"
TGT_LIST="./generation_img_splits/sd_v3_dataset_acdc_tr_snow_n400_rand1234.txt"
TRANS_DIR="/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_snow_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
SRC_LIST="./generation_img_splits/cityscapes_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/cityscapesDA/gtFine/train"
SRC_DATASET="cityscapes"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1024 2048 --src_dataset $SRC_DATASET  |& tee -a "metrics/$EXP_NAME.txt"

### Cityscapes --> ACDC NIGHT ###

EXP_NAME="CS_ACDC_NIGHT_Metrics_FID"
TGT_DIR="/srv/share4/datasets/ACDC/rgb_anon/night/train"
TGT_LIST="./generation_img_splits/sd_v3_dataset_acdc_tr_night_n400_rand1234.txt"
TRANS_DIR="/srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234/checkpoint_2k/cityscapes_acdc_tr_night_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_10/images"
SRC_DIR="/srv/share4/datasets/cityscapesDA/leftImg8bit/train"
SRC_LIST="./generation_img_splits/cityscapes_img_split_700.txt"
SRC_LBL_DIR="/srv/share4/datasets/cityscapesDA/gtFine/train"
SRC_DATASET="cityscapes"

python get_perceptual_distance.py --target_dir $TGT_DIR --target_img_list $TGT_LIST --trans_dir $TRANS_DIR --src_dir $SRC_DIR --src_img_list $SRC_LIST --src_label_dir $SRC_LBL_DIR --size 1024 2048 --src_dataset $SRC_DATASET  |& tee -a "metrics/$EXP_NAME.txt"

