#!/bin/bash
source ~/.bashrc

# activate diffusers conda env
conda activate diffusers-vv

# put your directory for diffusers repo
cd /coc/scratch/vvijaykumar6/diffusion-da/diffusers/examples/dreambooth/generation_img_splits


DATASET_NAME="synthia_trans_synthia"
ROOT_OUTPUT_DIR="/coc/flash9/vvijaykumar6/diffusionda/datasets"
SRC_DIR="/coc/flash9/datasets/Synthia/RAND_CITYSCAPES"
TRANS_SRC_DIR="/coc/flash9/prithvi/diffusion_DA/hres_fixed_full_dataset_translation/genrun_debug_rml_sd_v15_pp_city_n20_res_512_crop_512_2k_iters_rand1234/checkpoint_2k/synthia_city_imglist_n24966_use_segmap_1_use_edge_0.5_res1280_gen-seed_10"
IMG_DIR="RGB"
GT_DIR="GT/LABELS"
NUM_IMAGES=700
MIXING_RATIO=0.5
NO_OVERLAPPING=True
DATASET_NAME="synthia_translated_src_n$NUM_IMAGES-mixing_ratio_$MIXING_RATIO-no_overlapping_$NO_OVERLAPPING"



python mix_dataset.py \
 --dataset_name $DATASET_NAME \
 --root_output_dir $ROOT_OUTPUT_DIR \
 --src_dir $SRC_DIR \
 --trans_src_dir $TRANS_SRC_DIR \
 --img_dir $IMG_DIR \
 --num_images $NUM_IMAGES \
 --mixing_ratio $MIXING_RATIO \
 --no_overlapping $NO_OVERLAPPING


conda deactivate

# directory for MIC repo
cd /coc/scratch/vvijaykumar6/diffusion-da/MIC/seg

# activate conda env for MIC
conda activate diffusionda

python tools/convert_datasets/synthia.py $ROOT_OUTPUT_DIR/$DATASET_NAME --gt-dir labels --nproc 8







