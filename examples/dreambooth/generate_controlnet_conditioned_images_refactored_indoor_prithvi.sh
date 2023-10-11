# Generating Sample Dataset (Habitat Sim -> MP3D)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/prithvi/misc/habitat_showcase_data/sim_data/ \
        --src_imglist files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_mlsd 1 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/habitat_showcase_data/translated_data/tr_v2_seed_$SEED
done

# Generating Sample Dataset (ProcTHOR -> MP3D)
for SEED in 10 20 30
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/share4/prithvi/misc/procthor_data/sim_data/ \
        --src_imglist procthor_files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_mlsd 1 \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/procthor_data/translated_data/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> MP3D)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_seg_map 1 \
        --use_mlsd 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_mlsd0.5/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> MP3D)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_edge0.5/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> MP3D)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_seg_map 1 \
        --use_softedge 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_sedge0.5/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> CODA)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 640 \
        --use_seg_map 1 \
        --use_mlsd 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_mlsd0.5_to_coda/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> CODA)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 640 \
        --use_edge 1 \
        --use_mlsd 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_edge0.5_to_coda/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> CODA)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 640 \
        --use_seg_map 1 \
        --use_softedge 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_sedge0.5_to_coda/tr_seed_$SEED
done

# With Shuffle Guidance
# ***********************************************************************************************

# Generating Sample Dataset (HSSD -> MP3D)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath genrun_v1_rml_sd_v15_pp_habitat_real_n500_res_512_crop_512_rand1234 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 512 \
        --use_seg_map 1 \
        --use_edge 0.5 \
        --use_shuffle 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --read_im_inorder 1 \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_edge0.5_shuf0.5/tr_seed_$SEED
done

# Generating Sample Dataset (HSSD -> CODA)
# for SEED in 10 20 30
for SEED in 10
do
    python generate_controlnet_conditioned_images_refactored.py \
        --rootdir_filepath logs/checkpoints \
        --filepath testing_v6_rml_sd_v15_pp_coda_n350_rand1234_r640_c480 \
        --ckpt 2000 \
        --src_imgdir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/rgb/ \
        --src_lbldir /srv/flash2/syenamandra3/diff-da/datadump/images/hssd_trajs/1011/102816756_600/semantic/ \
        --src_imglist hssd_files.txt \
        --seed $SEED \
        --resolution 640 \
        --use_edge 1 \
        --use_mlsd 0.5 \
        --use_shuffle 0.5 \
        --base_dset hssd \
        --no_overwrite True \
        --read_im_inorder 1 \
        --save_dir /srv/share4/prithvi/misc/hssd_data/translated_data_w_seg1_edge0.5_shuf0.5_to_coda/tr_seed_$SEED
done