# ---------------------------------------------------------------
# Copyright 2023 Telecom Paris, Yasser BENIGMIM. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------

import os
import torch
import random
import itertools
import matplotlib.pyplot as plt
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
import argparse
from my_utils import create_directory, generate_list_ckpt


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--filepath",
        required=True,
        type=str,
        help="path of the model",
    )
    parser.add_argument(
        "--prompt",
        default="semantic segmentation of an urban street scene in sks",
        type=str,
        help="generation prompt",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=200,
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
    args, _ = parser.parse_known_args()

    return args

def main(args):
    prompt = args.prompt
    output_folder = os.path.join(
        "logs/images", args.filepath, f"{args.filepath}_e{args.ckpt}"
    )
    model_id = os.path.join("logs/checkpoints", args.filepath, f"inf_ckpt{args.ckpt}")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, torch_dtype=torch.float16
    ).to("cuda")
    create_directory(output_folder)
    for j in range(args.num_images):
        generator = torch.Generator(device="cpu").manual_seed(random.randint(0, 100))
        # image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, generator=generator).images[0]
        image = pipe(prompt, num_inference_steps=50, guidance_scale=7.5, width=1024, generator=generator).images[0]
        image.save(os.path.join(output_folder, f"im_{j}.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
