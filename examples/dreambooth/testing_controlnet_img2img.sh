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
        --img_dump_dir sdv15_ctv11_gta2real 
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