'''
BCGen Pipeline
Created on March 25, 2024
Last Update: April 1, 2024
Authors: Sina Davari, Ali Tohidifar
This code is for utilizing ControlNet to generate more realistic synthtic images.
'''
# Libraries
import cv2
import torch
import logging
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import LMSDiscreteScheduler, DDIMScheduler,DPMSolverMultistepScheduler, EulerDiscreteScheduler,PNDMScheduler,DDPMScheduler, EulerAncestralDiscreteScheduler, UniPCMultistepScheduler
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline
from utils import colorize_mask, avatarcutpaster

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bcgen(
        base_dataset_path='/home/sina/Downloads/Synthetic_Dataset_test',
        output_base_folder = './generated',
        output_img_size = (1280, 1280),
        num_inference_steps = 3, #80, # default = 50. The number of denoising steps
        controlnet_conditioning_scale =  [0.8, 0.8], # depth weight, segmentation weight 
        img_strength = 0.9,
        CFG = 12.5, # default = 7.5 — The higher CFG, more dependence on the text prompt at the expense of lower img quality.
        prompt = "a high quality, high resolution image of a construction site",
        n_prompts = "blurry, blurred, ugly, bad anatomy, low quality",
        cutpaste = True,
        ):
    
    logging.info("\nStarting BCGen Pipeline...\n")
    seed_num = 42

    # Instantiate ControlNet
    controlnet = [
                ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16),
                ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16),
                ]

    # other options:
    # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16),
    # ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16),

    pipe =  StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate data and save it to the folders
    generator = torch.Generator(device="cpu").manual_seed(seed_num)


    # Convert string paths to Path objects
    base_dataset_path = Path(base_dataset_path)
    output_base_folder = Path(output_base_folder)
    output_base_folder.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime('_%Y-%m-%d_%H-%M-%S')

    # List all subfolders in the base_dataset_path
    subfolders = [f for f in base_dataset_path.iterdir() if f.is_dir()]

    for img_seq_folder in subfolders:
        # Update the input_image_folder to point to the current subfolder
        input_image_folder = img_seq_folder

        img_list = list(input_image_folder.glob('*.jpg'))
        controlnet_input_images = []
        controlnet_conditions = []

        for img_path in img_list:
            # Updating paths to remove redundant input_image_folder
            img_dir = img_path
            mask_name = img_path.name.replace('test', 'Image')
            mask_dir = img_dir.parent / "Semantic Segmentation" / mask_name
            depth_dir = Path(str(mask_dir).replace("Semantic Segmentation", "Depth Map"))

            # load image
            image = Image.open(img_dir)
            image = image.resize(output_img_size, Image.ANTIALIAS)
            controlnet_input_images.append((img_path.name, image))

            # load depth and reverse it
            depth_image = Image.fromarray(255 - cv2.imread(str(depth_dir), cv2.IMREAD_UNCHANGED)).resize(output_img_size, Image.ANTIALIAS)
            
            # load mask
            mask_image = Image.open(mask_dir)
            mask = np.array(mask_image)
            if mask.ndim == 3: mask = mask[:, :, 0]
            
            # preprocess the masks
            mask_image = Image.fromarray(colorize_mask(mask)).resize(output_img_size, Image.ANTIALIAS)
            
            # add conditions to the list
            controlnet_conditions.append([depth_image, mask_image])

        folder_path = output_base_folder / img_seq_folder.name
        # print('\nProcessing folder "', folder_path, '"')
        logging.info(f'\nProcessing folder: {folder_path}')
        folder_path_with_timestamp = Path(str(folder_path) + timestamp)
        folder_path_with_timestamp.mkdir(parents=True, exist_ok=True)

        # Generate and save images
        for (img, image), condition_image in zip(controlnet_input_images, controlnet_conditions):
            out_image = pipe(
                prompt = prompt,
                negative_prompt = n_prompts,
                control_image = condition_image,
                image = image,
                height = output_img_size[0],
                width = output_img_size[1],
                guidance_scale = CFG,
                strength = img_strength,
                num_inference_steps=num_inference_steps,
                generator=generator,
                controlnet_conditioning_scale = controlnet_conditioning_scale,
                cutpaste = True
            ).images[0]
            
            # Save the image with a unique filename
            out_img_path = folder_path_with_timestamp / img
            out_image.save(out_img_path)

        # Avatar Cut and Paste
        if cutpaste:
            avatarcutpaster(Path.cwd() / input_image_folder / img_seq_folder, Path.cwd(), output_img_size, timestamp)
        
    logging.info("\nBCGen Pipeline executed successfully.")

if __name__ == "__main__":
    bcgen()