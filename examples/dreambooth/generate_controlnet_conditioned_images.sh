# Generating sample dataset
python generate_controlnet_conditioned_images.py \
    --filepath testing_v5_rml_sd_v15_pp_city_n100_rand1234 \
    --ckpt 13000 \
    --src_imgdir /srv/share4/datasets/GTA5DA/images/ \
    --src_lbldir /srv/share4/datasets/GTA5DA/labels/ \
    --src_imglist blah \
    --use_edge 1 \
    --save_dir /srv/share4/prithvi/diffusion_da/model_testing_v5_rml_sd_v15_pp_city_n100_rand1234_random_gtav2real_samples_use_edge_0.5_res1024_full_dataset_cty_size_2975_v6