import re
import os
import cv2
import json
import torch
import random
import requests
import argparse
import itertools

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from io import BytesIO
from pprint import pprint
from accelerate import Accelerator
from torchvision import transforms
from my_utils import create_directory, generate_list_ckpt
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler

random.seed(1234)

# Color-code Mapping from GTAV / Cityscapes to ADE20k
CITY2ADE_MAP = json.load(open("city2ade_translate_map.json", "r"))
NYU2ADE_MAP = json.load(open("nyu2ade_translate_map.json", "r"))


# N_IMGS = 50
N_IMGS = 2975

GTAV_IMIDS = ['{0:05d}'.format(x) + ".png" for x in random.sample(list(range(1, 24966)), N_IMGS)]

# Get Edge Image
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
def prep_images(imgpath, labelpath, USE_MAP=CITY2ADE_MAP, car_hood_fix=0, palette_quant=0):
    # Load image and label
    image = Image.open(imgpath).convert("RGB")
    lbl = Image.open(labelpath).convert("RGB")

    if car_hood_fix:
        lbl = replace_black_pixels_vectorized(lbl)
    if palette_quant == 1:
        lbl = gen_palette_quantization(USE_MAP, lbl)

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
    parser = argparse.ArgumentParser(description="Simple example of a generation script.")
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
        "--src_imgdir",
        required=True,
        type=str,
        help="path of the source image directory",
    )
    parser.add_argument(
        "--src_lbldir",
        required=True,
        type=str,
        help="path of the source image-label directory",
    )
    parser.add_argument(
        "--src_imglist",
        required=True,
        type=str,
        default=None,
        help="path of the source image list",
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        type=str,
        help="path to save directory",
    )
    parser.add_argument(
        "--use_edge",
        required=False,
        type=int,
        default=0,
        help="use edge conditioning?",
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
        "--num_steps",
        default=50,
        type=int,
        help="number of inference steps",
    )
    parser.add_argument(
        "--seed",
        default=10,
        type=int,
        help="Generation Seed",
    )
    # NEED TO USE
    parser.add_argument(
        "--palette_quant",
        default=0,
        type=int,
        help="Palette Quantization?",
    )
    # NEED TO USE
    parser.add_argument(
        "--car_hood_fix",
        default=0,
        type=int,
        help="Fix don't care label for car-hood",
    )
    args, _ = parser.parse_known_args()
    return args

def main(args):
    # Setup model and devices
    device = "cuda"
    
    if args.use_edge == 1:
        args.use_edge = True
    else:
        args.use_edge = False
        
    print("use edge", args.use_edge)
    
    # ControlNet
    if args.use_edge:
        controlnet = [
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16),
        ]
    else:
        controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16),
        
    # Base SD-v1.5 model
    if args.rootdir_filepath:
        model_id = os.path.join(args.rootdir_filepath, args.filepath, f"inf_ckpt{args.ckpt}")
    else:
        model_id = os.path.join("logs/checkpoints", args.filepath, f"inf_ckpt{args.ckpt}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.safety_checker = None
    # pipe.requires_safety_checker = False
    
    # Prompt + Generator
    if args.use_edge:
        # prompt = ["egocentric view of a high resolution, photorealistic urban street scene in sks"] * 2
        prompt = [args.prompt]*2
    else:
        # prompt = "egocentric view of a high resolution, photorealistic urban street scene in sks"
        prompt = args.prompt

    # set palette we use
    USE_MAP = CITY2ADE_MAP
    if args.base_dset == "nyu":
        USE_MAP = NYU2ADE_MAP

    GEN_SEED = args.seed
    generator = [torch.Generator(device="cpu").manual_seed(GEN_SEED) for i in range(len(prompt))]
    
    # Setup save folder
    output_folder = args.save_dir
    create_directory(output_folder)
    
    # Loop over imagelist
    
    # Number of denoising steps
    # N_STEPS = 20
    N_STEPS = args.num_steps
    # N_STEPS = 100
    
    # Resolution of generated image
    # RES = 768
    # RES = 1024
    RES = args.resolution
    
    if not args.src_imglist:
        print("Generating over entire dataset")
        args.src_imglist = GTAV_IMIDS
        imglist = args.src_imglist
    else:
        print("Using image list", args.src_imglist)
        with open(args.src_imglist, 'r') as file:
            imglist = file.read().splitlines() 
        args.src_imglist = imglist
    
    print("IMAGE LIST LENGTH",len(imglist))
    N_IMGS = len(imglist)
    
    # Save metadata
    metadata = {
        "prompt": prompt[0],
        "n_steps": N_STEPS,
        "n_images": N_IMGS,
        "input_res": RES,
        "generator_seed": GEN_SEED,
    }
    
    with open(os.path.join(output_folder, "metadata.json"), "w") as f:
        json.dump(metadata, f)
    
    for imgid in imglist:
        imgpath = os.path.join(args.src_imgdir, imgid)
        lblpath = os.path.join(args.src_lbldir, imgid)
        img, lbl, imsize = prep_images(imgpath, lblpath, USE_MAP=USE_MAP, car_hood_fix=args.car_hood_fix, palette_quant=args.palette_quant)
        img.thumbnail((RES, RES))
        lbl.thumbnail((RES, RES))
        if args.use_edge:
            edge_img = get_edge_image(img)
        post_transform = transforms.Resize(imsize[::-1])
        if args.use_edge:
            conditioning = [lbl, edge_img]
        if args.use_edge:
            image = pipe(
                prompt,
                conditioning,
                num_inference_steps=N_STEPS,
                generator=generator,
                controlnet_conditioning_scale=[1.0, 0.5],
            ).images[0]
        else:
            image = pipe(
                prompt,
                lbl,
                num_inference_steps=N_STEPS,
                generator=generator,
            ).images[0]
        sz = tuple(imsize[::-1])[::-1]
        image = image.resize(sz, Image.LANCZOS)
        savepath = os.path.join(output_folder, imgid)
        image.save(savepath)
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    