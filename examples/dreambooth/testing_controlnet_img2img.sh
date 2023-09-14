# Sample Generation Script Using ControlNet
python testing_controlnet_img2img.py \
    --filepath testing_proposal_v5_rml_sd_v15_pp_city_n100_rand1234 \
    --ckpt 13000 \
    --src_img /srv/share4/datasets/GTA5DA/images/00016.png \
    --src_lbl /srv/share4/datasets/GTA5DA/labels/00016.png \
    --img_id 16