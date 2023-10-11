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

# Convert Dark Zurich Checkpoints
# genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_dz_tr_all_n500_res_512_crop_512_rand1234
# genrun_v1_rml_sd_v15_pp_dz_tr_night_n500_res_512_crop_512_rand1234

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_dz_tr_all_n500_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_dz_tr_night_n500_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000

python convert_dreambooth.py \
    --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
    --max_train_steps 2000 \
    --checkpointing_steps 1000