# Generating sample dataset
# python generate_controlnet_conditioned_images.py \
#     --filepath testing_v5_rml_sd_v15_pp_city_n100_rand1234 \
#     --ckpt 13000 \
#     --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
#     --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
#     --src_imglist blah \
#     --use_edge 1 \
#     --save_dir /srv/share4/prithvi/diffusion_da/model_testing_v5_rml_sd_v15_pp_city_n100_rand1234_random_gtav2real_samples_use_edge_0.5_res1024_full_dataset_cty_size_2975_v6


# Generating sample dataset (gta --> cityscapes)
python generate_controlnet_conditioned_images_refactored.py \
    --rootdir_filepath /srv/share4/prithvi/diffusion_testing/checkpoints \
    --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
    --ckpt 2000 \
    --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
    --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
    --src_imglist generation_img_splits/gta_img_split_2.txt \
    --seed 10 \
    --resolution 1024 \
    --car_hood_fix 1 \
    --use_seg_map 1 \
    --use_edge 0.5 \
    --save_dir /srv/share4/vvijaykumar6/diffusion-da/datasets/testing_v6_rml_sd_v15_pp_city_n500_rand1234/checkpoint_2k/gta_cs_imglist_n2_use_segmap_1_use_edge_0.5_res1024_gen-seed_10