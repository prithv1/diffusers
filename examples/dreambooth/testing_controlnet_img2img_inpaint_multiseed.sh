for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img_inpaint.py \
            --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir inpaint_map_sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

python testing_controlnet_img2img_inpaint.py \
    --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
    --ckpt 2000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00087.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00087.png \
    --img_id 87 \
    --img_dump_dir inpaint_city_sdv15_ctv11_gta2real_v4_g50_c2k_seed30 \
    --seed 30 \
    --car_hood_fix 1 \
    --class_inpaint road,vegetation,sky \
    --inpaint_mode 1

python testing_controlnet_img2img_inpaint.py \
    --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
    --ckpt 2000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00087.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00087.png \
    --img_id 87 \
    --img_dump_dir inpaint_map_sdv15_ctv11_gta2real_v4_g50_c2k_seed30 \
    --seed 30 \
    --car_hood_fix 1 \
    --class_inpaint vegetation \
    --inpaint_mode 1 \
    --use_ft 0 \
    --prompt bright red trees in sks

python testing_controlnet_img2img_inpaint.py \
    --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
    --ckpt 2000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00087.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00087.png \
    --img_id 87 \
    --img_dump_dir inpaint_city_sdv15_ctv11_gta2real_v4_g50_c2k_seed30 \
    --seed 30 \
    --car_hood_fix 1 \
    --class_inpaint trees \
    --inpaint_mode 1 \
    --palette_quant 1 \
    --prompt bright red cars in sks

python testing_controlnet_img2img_inpaint.py \
    --filepath testing_v6_rml_sd_v15_pp_gtav_n500_rand1234 \
    --ckpt 2000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00087.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00087.png \
    --img_id 87 \
    --img_dump_dir inpaint_gtav_sdv15_ctv11_gta2real_v4_g50_c2k_seed30 \
    --seed 30 \
    --car_hood_fix 1 \
    --class_inpaint sky \
    --inpaint_mode 1 \
    --prompt "sunny sky, high resolution"