# ***************************** DARK ZURICH DATASET ***********************************************************************
############## Train Dreambooth Dark Zurich Train-All ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_dz_tr_all_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_dz_tr_all_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_dz_tr_all_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth Dark Zurich Train-Night ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_dz_tr_night_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_dz_tr_night_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_dz_tr_night_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth Dark Zurich Train-Day ########################
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_dz_tr_day_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_dz_tr_day_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_dz_tr_day_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth Dark Zurich Train-Twilight ########################
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_dz_tr_twilight_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_dz_tr_twilight_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_dz_tr_twilight_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

# ***************************** DARK ZURICH DATASET ***********************************************************************

# ***************************** ACDC DATASET ***********************************************************************
############## Train Dreambooth ACDC Train-All ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_acdc_tr_all_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_acdc_tr_all_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_acdc_tr_all_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth ACDC Train-Fog ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_acdc_tr_fog_n400_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_acdc_tr_fog_n400_res_512_crop_512_rand1234 \
    --num_class_images 400 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_acdc_tr_fog_n400_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth ACDC Train-Rain ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_acdc_tr_rain_n400_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_acdc_tr_rain_n400_res_512_crop_512_rand1234 \
    --num_class_images 400 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_acdc_tr_rain_n400_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth ACDC Train-Snow ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_acdc_tr_snow_n400_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_acdc_tr_snow_n400_res_512_crop_512_rand1234 \
    --num_class_images 400 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_acdc_tr_snow_n400_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

############## Train Dreambooth ACDC Train-Night ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_acdc_tr_night_n400_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_acdc_tr_night_n400_res_512_crop_512_rand1234 \
    --num_class_images 400 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_acdc_tr_night_n400_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1

# sd_v3_dataset_habitat_real_n500_rand1234
############## Train Dreambooth Habitat Indoor ######################## [Running]
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_habitat_real_n500_rand1234 \
    --output_dir genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_habitat_real_n500_rand1234 \
    --class_prompt "a high-resolution photorealistic real world indoor scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "a high-resolution photorealistic real world indoor scene in sks" \
    --max_train_steps 2000 \
    --resolution 512 \
    --crop-size 512 \
    --resize_reg_img 1