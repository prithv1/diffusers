# Sample Generation Script Using ControlNet

# With cityscapes finetuned
python testing_controlnet_img2img.py \
    --filepath testing_v5_rml_sd_v15_pp_city_n100_rand1234 \
    --ckpt 13000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00016.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00016.png \
    --img_id 16


# With mapillary finetuned
python testing_controlnet_img2img.py \
    --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
    --ckpt 13000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00016.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00016.png \
    --img_id 16

##################################################################################################################

# Off-the-Shelf Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
        --ckpt 13000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --use_ft 0 \
        --img_dump_dir sdv15_ctv11_gta2real_v4_g50 
done

# Mapillary Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
        --ckpt 13000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir map_sdv15_ctv11_gta2real 
done

# Cityscapes Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v5_rml_sd_v15_pp_city_n100_rand1234 \
        --ckpt 13000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir city_sdv15_ctv11_gta2real 
done

##################################################################################################################

# Mapillary Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir map_sdv15_ctv11_gta2real_v4_g50_c2k
done

# Cityscapes Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir city_sdv15_ctv11_gta2real_v4_g50_c2k
done

for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_city_n500_res_1024_crop_1024_rand1234 \
        --ckpt 1000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir city_r1024_c1024_sdv15_ctv11_gta2real_v4_g50_c1k
done

# NuScenes Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir nusc_sdv15_ctv11_gta2real_v4_g50_c2k
done

# Synthia Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir synthia_sdv15_ctv11_gta2syn_v4_g50_c2k
done

# BDD Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_bdd10k_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir bdd_sdv15_ctv11_gta2real_v4_g50_c2k
done

#################################################***NYUv2 Testing***####################################################
# # Off-the-Shelf Stable Diffusion
# for ID in 00 09 15 28 64 99
# do
#     python testing_controlnet_img2img.py \
#         --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
#         --ckpt 13000 \
#         --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_010_005/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
#         --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_010_005/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
#         --img_id $ID \
#         --use_ft 0 \
#         --img_dump_dir sdv15_ctv11_hsim2real \
#         --base_dset nyu \
#         --resolution 640 \
#         --prompt a photorealistic real world indoor scene in sks
# done

for ID in 00 13 33 65 88 99
do
    python testing_controlnet_img2img.py \
        --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
        --ckpt 13000 \
        --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
        --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
        --img_id $ID \
        --use_ft 0 \
        --img_dump_dir sdv15_ctv11_hsim2real_v4_g50_r1024 \
        --base_dset nyu \
        --resolution 1024 \
        --prompt a high-resolution photorealistic real world indoor scene in sks
done

for ID in 00 13 33 65 88 99
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234 \
        --ckpt 15000 \
        --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
        --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
        --img_id $ID \
        --img_dump_dir nyuv2_sdv15_ctv11_hsim2real_v4_g50_r640 \
        --base_dset nyu \
        --resolution 640 \
        --prompt a high-resolution photorealistic real world indoor scene in sks
done

for ID in 00 13 33 65 88 99
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234 \
        --ckpt 10000 \
        --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
        --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
        --img_id $ID \
        --img_dump_dir nyuv2_sdv15_ctv11_hsim2real_v4_g50_r640_c10k \
        --base_dset nyu \
        --resolution 640 \
        --prompt a high-resolution photorealistic real world indoor scene in sks
done

for ID in 00 13 33 65 88 99
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
        --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
        --img_id $ID \
        --img_dump_dir nyuv2_sdv15_r640_c480_ctv11_hsim2real_v4_g50_r640_c2k \
        --base_dset nyu \
        --resolution 640 \
        --prompt a high-resolution photorealistic real world indoor scene in sks
done


################### Generate From Existing at 768x768 Resolution
# Mapillary Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
        --ckpt 15000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir map_sdv15_ctv11_gta2real_res768 \
        --resolution 768
done

# Cityscapes Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
        --ckpt 15000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir city_sdv15_ctv11_gta2real_res768 \
        --resolution 768
done

# NuScenes Fine-tuned Stable Diffusion
for ID in 16 25 33 43 50
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
        --ckpt 15000 \
        --src_img /srv/share4/datasets/GTA5DA/images/000$ID.png \
        --src_lbl /srv/share4/datasets/GTA5DA/labels/000$ID.png \
        --img_id $ID \
        --img_dump_dir nusc_sdv15_ctv11_gta2real_res768 \
        --resolution 768
done

############ Generate Images from Generate Scene Layouts
# NuScenes Fine-tuned Stable Diffusion
for ID in 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
        --ckpt 2000 \
        --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234_e2000/im_$ID.png \
        --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/logs/images/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234/testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234_e2000/im_$ID.png \
        --img_id $ID \
        --img_dump_dir map_sdv15_ctv11_syngen_gta2real_res512 \
        --resolution 512 \
        --palette_quant 1
done

# /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/test_q.png
python testing_controlnet_img2img.py \
    --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
    --ckpt 2000 \
    --src_img /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/test_q.png \
    --src_lbl /nethome/prithvijit3/projects/Diffusion-DA/generation/diffusers/examples/dreambooth/test_q.png \
    --img_id 10 \
    --img_dump_dir map_sdv15_ctv11_syngen_gta2real_res512 \
    --resolution 512

########################## CODA Finetuned Testing

for ID in 00 13 33 65 88 99
do
    python testing_controlnet_img2img.py \
        --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
        --ckpt 2000 \
        --src_img /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_final_preview/frame.00$ID.color.jpg \
        --src_lbl /srv/share4/datasets/Hypersim/raw_dataset/ai_015_001/images/scene_cam_00_geometry_preview/frame.00$ID.semantic.png \
        --img_id $ID \
        --img_dump_dir coda_sdv15_r512_c512_ctv11_hsim2real_v4_g50_r512_c2k \
        --base_dset nyu \
        --resolution 512 \
        --prompt a high-resolution photorealistic real world indoor scene in sks
done