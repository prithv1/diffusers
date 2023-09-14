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