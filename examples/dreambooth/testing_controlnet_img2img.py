from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionImg2ImgPipeline, UniPCMultistepScheduler
import re
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

import cv2
import numpy as np

from pprint import pprint

# Need to convert cityscapes colors to ADE20K

CITY2ADE_MAP = json.load(open("city2ade_translate_map.json", "r"))

def get_edge_image(image):
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
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
    args, _ = parser.parse_known_args()

    return args

def main(args):
    device = "cuda"
    output_folder = os.path.join(
            "logs/images", args.filepath, f"ctnet_i2i_debug_{args.filepath}_e{args.ckpt}"
        )
    
    # We'll condition on both segmentation and image
    controlnet = [
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_seg", torch_dtype=torch.float16),
        ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_canny", torch_dtype=torch.float16),
    ]
    print(output_folder)
    create_directory(output_folder)
    
    # To work with finetuned checkpoint
    model_id = os.path.join("logs/checkpoints", args.filepath, f"inf_ckpt{args.ckpt}")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_id, controlnet=controlnet, torch_dtype=torch.float16
    )
    
    # To work with off-the-shelf checkpoint
    # pipe = StableDiffusionControlNetPipeline.from_pretrained(
    #     "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    # )
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload() 
    # pipe.enable_xformers_memory_efficient_attention()
    
    prompt = ["egocentric view of a high resolution, photorealistic urban street scene in sks"]*2
    generator = [torch.Generator(device="cpu").manual_seed(25) for i in range(len(prompt))]
    
    init_image = Image.open(args.src_img).convert("RGB")
    init_lbl = Image.open(args.src_lbl).convert("RGB")
    init_lbl.save(os.path.join(output_folder, f"src_orig_lbl_{str(args.img_id)}.png"))
    
    # Preprocess label
    init_lbl_arr = np.array(init_lbl)
    init_lbl_shape = init_lbl_arr.shape
    init_lbl_arr = init_lbl_arr.reshape(-1, 3)
    unique_img_vals = [tuple(x) for x in list(np.unique(init_lbl_arr, axis=0))]
    conv_keys = list(CITY2ADE_MAP.keys())
    conv_keys = [tuple([int(x) for x in re.findall(r'\d+', k)]) for k in conv_keys]
    for uniq_val in unique_img_vals:
        key_idx = np.where(np.equal(init_lbl_arr, list(uniq_val)).all(1))[0]
        if uniq_val in conv_keys:
            init_lbl_arr[key_idx] = CITY2ADE_MAP[str(tuple(uniq_val))]
        else:
            init_lbl_arr[key_idx] = (0, 0, 0)
  
    init_lbl_arr = init_lbl_arr.reshape(init_lbl_shape)
    init_lbl = Image.fromarray(np.uint8(init_lbl_arr), "RGB")
    
    post_transform = transforms.Resize((1024, 1024))
    init_image.thumbnail((1024, 1024))
    edge_image = get_edge_image(init_image)
    init_lbl.thumbnail((1024, 1024))
    
    init_image.save(os.path.join(output_folder, f"src_im_{str(args.img_id)}.png"))
    init_lbl.save(os.path.join(output_folder, f"src_lbl_{str(args.img_id)}.png"))
    
    # Actual denoising generation process
    image = pipe(
        prompt,
        [init_lbl, edge_image],
        num_inference_steps=20, # Controllable hparam
        generator=generator,
        controlnet_conditioning_scale=[1.0, 0.5], # Controllable hparam
    ).images[0]
    
    image = image.resize((1914, 1052), Image.LANCZOS)
    image.save(os.path.join(output_folder, f"tr_im_{str(args.img_id)}.png"))
    
if __name__ == "__main__":
    args = parse_args()
    main(args)