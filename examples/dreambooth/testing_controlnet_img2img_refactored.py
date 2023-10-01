from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
import re
import ast
import json
import torch
import requests
from PIL import Image
from io import BytesIO

from torchvision import transforms
import os
import itertools
import matplotlib.pyplot as plt
from accelerate import Accelerator
import argparse
from my_utils import create_directory, generate_list_ckpt
from controlnet_aux import MLSDdetector

import cv2
import numpy as np

from pprint import pprint

from sklearn.metrics.pairwise import euclidean_distances

# mapping to ade20k
CITY2ADE_MAP = json.load(open("city2ade_translate_map.json", "r"))
NYU2ADE_MAP = json.load(open("nyu2ade_translate_map.json", "r"))

mlsd = MLSDdetector.from_pretrained('lllyasviel/ControlNet')

def get_edge_image(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

# Function to quantize generated scene layouts based on shared class palette
def gen_palette_quantization(palette_data, img):
    img_arr = np.array(img)
    img_shape = img_arr.shape
    img_arr = img_arr.reshape(-1, 3)
    
    palette = list(palette_data.keys())
    palette = [list(ast.literal_eval(x)) for x in palette]
    palette_arr = np.array(palette)
    
    distances = euclidean_distances(img_arr, palette_arr)
    closest_rows = np.argmin(distances, axis=1)
    img_arr[:] = palette_arr[closest_rows, :]
    img_arr = img_arr.reshape(img_shape)
    
    return Image.fromarray(img_arr)
    
# Function to fix the hood of the car issue
def replace_black_pixels_vectorized(img, height=400, replacement_color=(0, 0, 142)):
    img_array = np.array(img)

    # Extract R, G, B channels
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]

    # Create a mask for black pixels below the specified height
    black_pixels = (r == 0) & (g == 0) & (b == 0) & (np.arange(img_array.shape[0])[:, None] > height)

    # Apply replacement color to the selected pixels
    img_array[black_pixels] = replacement_color

    modified_img = Image.fromarray(np.uint8(img_array))
    return modified_img

# Function to prep images
def prep_images(imgpath, labelpath, output_folder, args, USE_MAP=CITY2ADE_MAP, car_hood_fix=0, palette_quant=0):
    # Load image and label
    image = Image.open(imgpath).convert("RGB")
    lbl = Image.open(labelpath).convert("RGB")

    if car_hood_fix:
        lbl = replace_black_pixels_vectorized(lbl)
    if palette_quant == 1:
        lbl = gen_palette_quantization(USE_MAP, lbl)
    
    lbl.save(os.path.join(output_folder, f"src_orig_lbl_{str(args.img_id)}.png"))

    # Get image size
    imsize = image.size

    # Label Translation
    init_lbl_arr = np.array(lbl)
    init_lbl_shape = init_lbl_arr.shape
    init_lbl_arr = init_lbl_arr.reshape(-1, 3)
    unique_img_vals = [tuple(x) for x in list(np.unique(init_lbl_arr, axis=0))]
    conv_keys = list(USE_MAP.keys())
    conv_keys = [tuple([int(x) for x in re.findall(r'\d+', k)]) for k in conv_keys]
    for uniq_val in unique_img_vals:
        key_idx = np.where(np.equal(init_lbl_arr, list(uniq_val)).all(1))[0]
        if uniq_val in conv_keys:
            init_lbl_arr[key_idx] = CITY2ADE_MAP[str(tuple(uniq_val))]
        else:
            init_lbl_arr[key_idx] = (0, 0, 0)
  
    init_lbl_arr = init_lbl_arr.reshape(init_lbl_shape)
    lbl = Image.fromarray(np.uint8(init_lbl_arr), "RGB")
    
    return image, lbl, imsize

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--rootdir_filepath",
        required=False,
        type=str,
        default=None,
        help="root dir of model",
    )
    parser.add_argument(
        "--filepath",
        required=True,
        type=str,
        help="path of the model",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--src_img",
        required=True,
        type=str,
        help="path of the source image",
    )
    parser.add_argument(
        "--src_lbl",
        required=True,
        type=str,
        help="path of the source label",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=2975,
        help="number of images to generate",
    )
    parser.add_argument(
        "--no-pretraining",
        action="store_true",
    )
    parser.add_argument(
        "--img_id",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--use_ft",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--rootdir_img_dump_dir",
        default=None,
        type=str,
        help="path of root dir to dump image",
    )
    parser.add_argument(
        "--img_dump_dir",
        default=None,
        type=str,
        help="path of folder to dump image",
    )
    parser.add_argument(
        "--prompt",
        default="egocentric view of a high resolution, photorealistic urban street scene in sks",
        type=str,
        help="text prompt for generation",
    )
    parser.add_argument(
        "--base_dset",
        default="cityscapes",
        type=str,
        help="Base dataset palette specs",
    )
    parser.add_argument(
        "--resolution",
        default=1024,
        type=int,
        help="Generation Resolution",
    )
    parser.add_argument(
        "--seed",
        default=25,
        type=int,
        help="Generation Seed",
    )
    parser.add_argument(
        "--palette_quant",
        default=0,
        type=int,
        help="Palette Quantization?",
    )
    
    parser.add_argument(
        "--car_hood_fix",
        default=0,
        type=int,
        help="Fix don't care label for car-hood",
    )

    parser.add_argument(
        "--use_seg_map",
        default=0,
        type=float,
        help="Use segmentation map w/ conidtioning param val specified",
    )

    parser.add_argument(
        "--use_edge",
        default=0,
        type=float,
        help="Use canny edge w/ conidtioning param val specified",
    )
    
    parser.add_argument(
        "--use_mlsd",
        default=0,
        type=float,
        help="Use straight line detector w/ conidtioning param val specified",
    )
    
    args, _ = parser.parse_known_args()

    return args

def main(args):
    device = "cuda"

    # ensure we have an input to controlnet
    # ORDER: segmap, edge, mlsd
    assert args.use_seg_map == 0 or args.use_edge == 0  or args.use_mlsd == 0, "At least one input to controlnet should be set."

    ############################
    ### CREATE OUTPUT FOLDER ###
    ############################
    if args.img_dump_dir is None:
        args.img_dump_dir = args.filepath
    
    if args.rootdir_img_dump_dir:
        output_folder = os.path.join(
            args.rootdir_img_dump_dir, args.img_dump_dir, f"ctnet_i2i_debug_e{args.ckpt}"
        )
    else:
        output_folder = os.path.join(
                "logs/images", args.img_dump_dir, f"ctnet_i2i_debug_e{args.ckpt}"
            )

    print(output_folder)
    create_directory(output_folder)
    
    ##########################################
    ### INIT CONTROL NET + STABLE DIFFUSION ##
    ##########################################

    # We'll condition on both segmentation and image

    controlnet = []

    print("Control Net inputs:")
    if args.use_seg_map:
        print("Using Seg Map")
        controlnet.append(ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16))
    if args.use_edge:
        print("Using Edge Image")
        controlnet.append(ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16))
    if args.use_mlsd:
        print("Using MLSD Image")
        controlnet.append(mlsd)

    # To work with finetuned checkpoint
    if args.use_ft == 1:
        
        if args.rootdir_filepath:
            model_id = os.path.join(args.rootdir_filepath, args.filepath, f"inf_ckpt{args.ckpt}")
        else:
            model_id = os.path.join("logs/checkpoints", args.filepath, f"inf_ckpt{args.ckpt}")
    
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            model_id, controlnet=controlnet, torch_dtype=torch.float16
        )
    else:
    # To work with off-the-shelf checkpoint
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
        )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() 
    # pipe.enable_xformers_memory_efficient_attention()

    # have as many prompts as we have for input to controlnet
    prompt = [args.prompt]*len(controlnet)
    generator = [torch.Generator(device="cpu").manual_seed(args.seed) for i in range(len(prompt))]

    #####################################
    ### IMAGE AND LABEL PREPROCESSING ###
    #####################################

    USE_MAP = CITY2ADE_MAP
    if args.base_dset == "nyu":
        USE_MAP = NYU2ADE_MAP

    init_image, init_lbl, imsize = prep_images(imgpath=args.src_img, labelpath=args.src_lbl, output_folder=output_folder, args=args, USE_MAP=USE_MAP, car_hood_fix=args.car_hood_fix, palette_quant=args.palette_quant)
    
    # transforms
    RES = args.resolution

    # TODO: fill in pre and post transforms
    pre_transform = None
    post_transform = transforms.Resize((RES, RES)) # currently not used

    #resize image, label, and edge image (thumbnail preserves image aspect ratio)
    init_image.thumbnail((RES, RES))
    init_lbl.thumbnail((RES, RES))

    # generate edge image
    if args.use_edge:
        edge_image = get_edge_image(init_image)
        edge_image.save(os.path.join(output_folder, f"src_edge_{str(args.img_id)}.png"))

    # generate mlsd image
    if args.use_mlsd:
        mlsd_image = mlsd(init_image)
        mlsd_image.thumbnail((RES, RES))
        mlsd_image.save(os.path.join(output_folder, f"src_mlsd_{str(args.img_id)}.png"))
    
    # save the pre-processed images
    init_image.save(os.path.join(output_folder, f"src_im_{str(args.img_id)}.png"))
    init_lbl.save(os.path.join(output_folder, f"src_lbl_{str(args.img_id)}.png"))

    
    # set images and conditioning scale for generation
    image_list = []

    # suggested values: 1, 0.5, 0.7
    controlnet_conditioning_scale = []

    if args.use_seg_map:
        image_list.append(init_lbl)
        controlnet_conditioning_scale.append(args.use_seg_map)
    if args.use_edge:
        image_list.append(edge_image)
        controlnet_conditioning_scale.append(args.use_edge)
    if args.use_mlsd == 1:
        image_list.append(mlsd_image)
        controlnet_conditioning_scale.append(args.use_mlsd)

    
    ##########################
    ### GENERATION PROCESS ###
    ##########################

    # Actual denoising generation process
    image = pipe(
        prompt,
        image_list,
        num_inference_steps=50, # Controllable hparam
        generator=generator,
        controlnet_conditioning_scale=controlnet_conditioning_scale, # Controllable hparam
    ).images[0]
    
    # image = image.resize((1914, 1052), Image.LANCZOS)
    image.save(os.path.join(output_folder, f"tr_im_{str(args.img_id)}.png"))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)