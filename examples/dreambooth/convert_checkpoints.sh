# Script to convert checkpoints for further processing

# For cityscapes finetuned dreambooth
python convert_dreambooth.py \
    --filepath testing_v5_rml_sd_v15_pp_city_n100_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

# For mapillary finetuned dreambooth
python convert_dreambooth.py \
    --filepath testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

############################################################################
python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

############################################################################
python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_gtav_semseg_n500_rand1234 \
    --max_train_steps 15000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234_r640_c480 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_bdd10k_n500_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_city_n500_res_1024_crop_1024_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_gtav_n500_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath testing_v6_rml_sd_v15_pp_hm3d_n500_rand1234_r300_c300 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000