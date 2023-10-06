
# Cityscapes Fine-tuned Stable Diffusion
for ID in 16 25 33
do
    python testing_controlnet_img2img_refactored.py \
        --rootdir_filepath /srv/share4/prithvi/diffusion_testing/checkpoints \
        --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --rootdir_img_dump_dir /srv/share4/vvijaykumar6/diffusion-da/logs/images \
        --img_dump_dir city_sdv15_ctetv11_gta2real_res1024_fix_car_refactored_only_seg \
        --resolution 1024 \
        --car_hood_fix 1 \
        --use_seg_map 1.0
        --use_edge 0.5
done