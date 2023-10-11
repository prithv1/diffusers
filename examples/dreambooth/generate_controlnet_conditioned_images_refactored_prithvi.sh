# Generating sample dataset (GTAV --> ACDC Night)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234/checkpoint_2k/gta_acdc_tr_night_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_$SEED
done

# Generating sample dataset (GTAV --> ACDC Snow)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234/checkpoint_2k/gta_acdc_tr_snow_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_$SEED
done

# Generating sample dataset (GTAV --> ACDC Rain)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234/checkpoint_2k/gta_acdc_tr_rain_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_$SEED
done

# Generating sample dataset (GTAV --> ACDC Fog)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234/checkpoint_2k/gta_acdc_tr_fog_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_$SEED
done

# Generating sample dataset (GTAV --> ACDC All)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234/checkpoint_2k/gta_acdc_tr_all_imglist_n700_use_segmap_1_use_edge_0.5_res1024_gen-seed_$SEED
done

for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
        --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
        --src_imglist generation_img_splits/gta_img_split_700.txt \
        --seed $SEED \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/diffusion_da_datasets/testing
done