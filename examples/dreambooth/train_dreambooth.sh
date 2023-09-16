# Script to finetune Dreambooth

# On Cityscapes
# python train_dreambooth.py \
#     --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
#     --instance_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
#     --output_dir testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
#     --num_class_images 100 \
#     --with_prior_preservation \
#     --class_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
#     --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --checkpointing_steps 1000 \
#     --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --max_train_steps 15000

python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_city_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_city_n500_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_city_n500_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 15000

python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_city_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_city_n500_res_1024_crop_1024_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_city_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 1024 \
    --crop-size 1024

# On Mapillary
# python train_dreambooth.py \
#     --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
#     --instance_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
#     --output_dir testing_v5_rml_sd_v15_pp_map_n100_rand1234 \
#     --num_class_images 100 \
#     --with_prior_preservation \
#     --class_data_dir /srv/share4/prithvi/sd_finetuning_datasets/sd_v2_dataset_mapillary_n100_rand1234 \
#     --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --checkpointing_steps 1000 \
#     --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --max_train_steps 15000

python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_mapillary_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_map_n500_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_mapillary_n500_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 15000

python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_mapillary_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_map_n500_res_1024_crop_1024_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_mapillary_n500_rand1234 \
    --class_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a high-resolution photorealistic urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 1024 \
    --crop-size 1024

# On NuScenes
# python train_dreambooth.py \
#     --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
#     --instance_data_dir sd_datasets/sd_v2_dataset_nuscenes_n100_rand1234 \
#     --output_dir testing_v5_rml_sd_v15_pp_nusc_n100_rand1234 \
#     --num_class_images 100 \
#     --with_prior_preservation \
#     --class_data_dir sd_datasets/sd_v2_dataset_nuscenes_n100_rand1234 \
#     --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --checkpointing_steps 1000 \
#     --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
#     --max_train_steps 15000

python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_nuscenes_n400_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_nusc_n400_rand1234 \
    --num_class_images 400 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_nuscenes_n400_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 15000

# On Synthia
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_synthia_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_synthia_n500_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_synthia_n500_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 2000

# On BDD10K
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_bdd10k_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_bdd10k_n500_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_bdd10k_n500_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 2000

# On GTAV
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_gtav_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_gtav_n500_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_gtav_n500_rand1234 \
    --class_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "egocentric view of a photorealistic urban street scene in sks" \
    --max_train_steps 2000

# On NYUv2
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_nyuv2_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_nyuv2_n500_rand1234_r640_c480 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_nyuv2_n500_rand1234 \
    --class_prompt "a photorealistic real world indoor scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "a photorealistic real world indoor scene in sks" \
    --max_train_steps 2000 \
    --resize_reg_img 1 \
    --resolution 640 \
    --crop-size 480

# On GTAV Annotations
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_gtav_semseg_n500_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_gtav_semseg_n500_res_1024_rand1234 \
    --num_class_images 500 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_gtav_semseg_n500_rand1234 \
    --class_prompt "semantic segmentation of an urban street scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "semantic segmentation of an urban street scene in sks" \
    --max_train_steps 2000 \
    --resolution 1024 \
    --crop-size 1024

# On CODA
python train_dreambooth.py \
    --pretrained_model_name_or_path "runwayml/stable-diffusion-v1-5" \
    --instance_data_dir sd_datasets/sd_v3_dataset_coda_n350_rand1234 \
    --output_dir testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
    --num_class_images 350 \
    --with_prior_preservation \
    --class_data_dir sd_datasets/sd_v3_dataset_coda_n350_rand1234 \
    --class_prompt "a high-resolution photorealistic real world indoor scene in sks" \
    --checkpointing_steps 1000 \
    --instance_prompt "a high-resolution photorealistic real world indoor scene in sks" \
    --max_train_steps 2000 \
    --resize_reg_img 1 \
    --resolution 512 \
    --crop-size 512