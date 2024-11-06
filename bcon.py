'''
BCon Pipeline
Created on March 25, 2024
Last Update: November 6, 2024
Authors: Sina Davari, Ali Tohidifar
Please see https://github.com/SinaDavari/bcon for more information.
'''
# Libraries
import cv2
import torch
import logging
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from diffusers import DDIMScheduler
from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionXLControlNetImg2ImgPipeline, AutoencoderKL
from utils import colorize_mask, avatarcutpaster

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def bcon(
        base_dataset_path='Path_to_the_synthetic_dataset',
        output_base_folder = 'Path_to_the_generated_dataset',
        output_img_size = (1280, 1280),
        num_inference_steps = 50,
        controlnet_conditioning_scale = [0.9, 0.9], # depth weight, segmentation weight 
        img_strength = 1,
        CFG = 12, # The higher CFG, more dependence on the text prompt at the expense of lower img quality.
        prompt = "a high quality, high resolution image of a construction site",
        n_prompts = "blurry, blurred, ugly, bad anatomy, bad quality, low quality",
        solver = DDIMScheduler,
        cutpaste = True,
        usexl = True,
        gridsearch = False,
        ):
    
    logging.info("\nStarting BCon Pipeline...\n")

    # Instantiate ControlNet
    if usexl:
        controlnet = [
            ControlNetModel.from_pretrained("diffusers/controlnet-depth-sdxl-1.0", variant="fp16", use_safetensors=True, torch_dtype=torch.float16).to("cuda"),
            ControlNetModel.from_pretrained("SargeZT/sdxl-controlnet-seg", torch_dtype=torch.float16).to("cuda"),
            ]
        
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16).to("cuda")

        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=controlnet,
            vae=vae,
            variant="fp16",
            use_safetensors=True,
            torch_dtype=torch.float16
        ).to("cuda")

    else:
        controlnet = [
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16),
            ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-seg", torch_dtype=torch.float16),
            ]

        pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            controlnet=controlnet,
            safety_checker=None,
            torch_dtype=torch.float16
        )

    pipe.scheduler = solver.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    # Generate data and save it to the folders
    generator = torch.Generator(device="cpu").manual_seed(42)

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
            image = image.resize(output_img_size, Image.LANCZOS)
            controlnet_input_images.append((img_path.name, image))

            # load depth and reverse it
            depth_image = Image.fromarray(255 - cv2.imread(str(depth_dir), cv2.IMREAD_UNCHANGED)).resize(output_img_size, Image.LANCZOS)
            
            # load mask
            mask_image = Image.open(mask_dir)
            mask = np.array(mask_image)
            if mask.ndim == 3: mask = mask[:, :, 0]
            
            # preprocess the masks
            mask_image = Image.fromarray(colorize_mask(mask)).resize(output_img_size, Image.LANCZOS)
            
            # add conditions to the list
            controlnet_conditions.append([depth_image, mask_image])

        ##### For Grid Search #####
        if gridsearch:
            folder_name = f"Steps_{num_inference_steps}-Cond_Scale_{controlnet_conditioning_scale[0]}_{controlnet_conditioning_scale[1]}-Strength_{img_strength}-CFG_{CFG}-Solver_{solver.__name__}_"
            folder_path = output_base_folder / Path(folder_name + img_seq_folder.name)
        else:
            folder_path = output_base_folder / img_seq_folder.name
        
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
                guidance_scale = CFG,
                strength = img_strength,
                num_inference_steps = num_inference_steps,
                generator = generator,
                controlnet_conditioning_scale = controlnet_conditioning_scale,
                solver = solver,
            ).images[0]
            
            # Save the image with a unique filename
            out_img_path = folder_path_with_timestamp / img
            out_image.save(out_img_path)

            torch.cuda.empty_cache()

        # Avatar Cut and Paste
        if cutpaste:
            avatarcutpaster(Path.cwd() / input_image_folder / img_seq_folder, Path.cwd(), folder_path_with_timestamp, output_img_size, timestamp)
        
    logging.info("\nBCon Pipeline executed successfully.")

if __name__ == "__main__":
    bcon()