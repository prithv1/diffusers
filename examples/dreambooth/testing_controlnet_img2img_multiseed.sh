##################################################################################################################

# Off-the-shelf Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --use_ft 0 \
            --img_dump_dir sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Mapillary Fine-tuned Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir map_sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Cityscapes Fine-tuned Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir city_sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# NuScenes Fine-tuned Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir nusc_sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Synthia Fine-tuned Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir synthia_sdv15_ctv11_gta2syn_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# BDD Fine-tuned Stable Diffusion
for SEED in 10 20 30 40
do
    for ID in 16 25 33 43 50
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_bdd10k_n500_rand1234 \
            --ckpt 2000 \
            --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
            --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
            --img_id $ID \
            --img_dump_dir bdd_sdv15_ctv11_gta2real_v4_g50_c2k_seed$SEED \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# # /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_final_preview
# # CODA Fine-tuned Stable Diffusion
# for SEED in 10 20 30 40
# do
#     for ID in 00 13 33 65 88 99
#     do
#         python testing_controlnet_img2img.py \
#             --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
#             --ckpt 2000 \
#             --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_final_preview/frame.00$ID.color.jpg \
#             --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_geometry_preview/frame.00$ID.semantic.png \
#             --img_id $ID \
#             --img_dump_dir coda_sdv15_r512_c512_ctv11_hsim2real_v4_g50_r512_c2k_seed$SEED \
#             --base_dset nyu \
#             --resolution 512 \
#             --prompt a high-resolution photorealistic real world indoor scene in sks \
#             --seed $SEED
#     done
# done

# for SEED in 10 20 30 40
# do
#     for ID in 00 13 33 65 88 99
#     do
#         python testing_controlnet_img2img.py \
#             --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
#             --ckpt 2000 \
#             --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_final_preview/frame.00$ID.color.jpg \
#             --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_geometry_preview/frame.00$ID.semantic.png \
#             --img_id $ID \
#             --img_dump_dir coda_sdv15_r512_c512_ctv11_hsim2real_v4_g50_r1024_c2k_seed$SEED \
#             --base_dset nyu \
#             --resolution 1024 \
#             --prompt a high-resolution photorealistic real world indoor scene in sks \
#             --seed $SEED
#     done
# done

# for SEED in 10 20 30 40
# do
#     for ID in 00 13 33 65 88 99
#     do
#         python testing_controlnet_img2img.py \
#             --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
#             --ckpt 2000 \
#             --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_final_preview/frame.00$ID.color.jpg \
#             --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_054_010/images/scene_cam_01_geometry_preview/frame.00$ID.semantic.png \
#             --img_id $ID \
#             --img_dump_dir coda_sdv15_r512_c512_ctv11_hsim2real_v4_g50_r640_c2k_seed$SEED \
#             --base_dset nyu \
#             --resolution 640 \
#             --prompt a high-resolution photorealistic real world indoor scene in sks \
#             --seed $SEED
#     done
# done

# GTAV Fine-tuned Stable Diffusion
for CITY in zurich hamburg
do
    for SEED in 10 20 30 40
    do
        for ID in 16 25 33 43 50
        do
            python testing_controlnet_img2img.py \
                --filepath testing_v6_rml_sd_v15_pp_gtav_n500_rand1234 \
                --ckpt 2000 \
                --src_img /srv/share4/datasets/cityscapesDA/leftImg8bit/train/"$CITY"/"$CITY"_0000"$ID"_000019_leftImg8bit.png \
                --src_lbl /srv/share4/datasets/cityscapesDA/gtFine/train/"$CITY"/"$CITY"_0000"$ID"_000019_gtFine_color.png \
                --img_id $ID \
                --img_dump_dir gtav_sdv15_ctv11_city2sim_v4_g50_c2k_"$CITY"_seed"$SEED" \
                --seed $SEED \
                --car_hood_fix 1
        done
    done
done

# Synthia Fine-tuned Stable Diffusion
for CITY in zurich erfurt
do
    for SEED in 10 20 30 40
    do
        for ID in 16 25 33 43 50
        do
            python testing_controlnet_img2img.py \
                --filepath testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
                --ckpt 2000 \
                --src_img /srv/share4/datasets/cityscapesDA/leftImg8bit/train/"$CITY"/"$CITY"_0000"$ID"_000019_leftImg8bit.png \
                --src_lbl /srv/share4/datasets/cityscapesDA/gtFine/train/"$CITY"/"$CITY"_0000"$ID"_000019_gtFine_color.png \
                --img_id $ID \
                --img_dump_dir synthia_sdv15_ctv11_city2sim_v4_g50_c2k_"$CITY"_seed"$SEED" \
                --seed $SEED \
                --car_hood_fix 1
        done
    done
done

# Cityscapes to Mapillary
for CITY in zurich erfurt
do
    for SEED in 10 20 30 40
    do
        for ID in 16 25 33 43 50
        do
            python testing_controlnet_img2img.py \
                --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
                --ckpt 2000 \
                --src_img /srv/share4/datasets/cityscapesDA/leftImg8bit/train/"$CITY"/"$CITY"_0000"$ID"_000019_leftImg8bit.png \
                --src_lbl /srv/share4/datasets/cityscapesDA/gtFine/train/"$CITY"/"$CITY"_0000"$ID"_000019_gtFine_color.png \
                --img_id $ID \
                --img_dump_dir map_sdv15_ctv11_city2real_v4_g50_c2k_"$CITY"_seed"$SEED" \
                --seed $SEED \
                --car_hood_fix 1
        done
    done
done

# Cityscapes to NuScenes
for CITY in zurich erfurt
do
    for SEED in 10 20 30 40
    do
        for ID in 16 25 33 43 50
        do
            python testing_controlnet_img2img.py \
                --filepath testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
                --ckpt 2000 \
                --src_img /srv/share4/datasets/cityscapesDA/leftImg8bit/train/"$CITY"/"$CITY"_0000"$ID"_000019_leftImg8bit.png \
                --src_lbl /srv/share4/datasets/cityscapesDA/gtFine/train/"$CITY"/"$CITY"_0000"$ID"_000019_gtFine_color.png \
                --img_id $ID \
                --img_dump_dir nusc_sdv15_ctv11_city2real_v4_g50_c2k_"$CITY"_seed"$SEED" \
                --seed $SEED \
                --car_hood_fix 1
        done
    done
done

# Generate Images from Pre-decided Layouts in Mapillary Style
for SEED in 10 20 30 40
do
    for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
            --ckpt 2000 \
            --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --img_id $ID \
            --img_dump_dir map_sdv15_ctv11_syngen_gta2real_res1024_seed"$SEED" \
            --resolution 1024 \
            --palette_quant 1 \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Generate Images from Pre-decided Layouts in Cityscapes Style
for SEED in 10 20 30 40
do
    for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
            --ckpt 2000 \
            --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --img_id $ID \
            --img_dump_dir city_sdv15_ctv11_syngen_gta2real_res1024_seed"$SEED" \
            --resolution 1024 \
            --palette_quant 1 \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Generate Images from Pre-decided Layouts in GTAV Style
for SEED in 10 20 30 40
do
    for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_gtav_n500_rand1234 \
            --ckpt 2000 \
            --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --img_id $ID \
            --img_dump_dir gtav_sdv15_ctv11_syngen_gta2real_res1024_seed"$SEED" \
            --resolution 1024 \
            --palette_quant 1 \
            --seed $SEED \
            --car_hood_fix 1
    done
done

# Generate Images from Pre-decided Layouts in Synthia Style
for SEED in 10 20 30 40
do
    for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
    do
        python testing_controlnet_img2img.py \
            --filepath testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
            --ckpt 2000 \
            --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234_e2000/im_$ID.png \
            --img_id $ID \
            --img_dump_dir synthia_sdv15_ctv11_syngen_gta2real_res1024_seed"$SEED" \
            --resolution 1024 \
            --palette_quant 1 \
            --seed $SEED \
            --car_hood_fix 1
    done
done