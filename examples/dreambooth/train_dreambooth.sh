# Script to finetune Dreambooth

# On Cityscapes
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
    --output_dir testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
    --num_class_images 100 \
    --with_prior_preservation \
    --class_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 15000

# On Mapillary
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
    --output_dir testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
    --num_class_images 100 \
    --with_prior_preservation \
    --class_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 15000