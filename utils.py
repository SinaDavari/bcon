import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Colorizing Masks 

def colorize_mask(gray_mask, background_threshold = 25):
    """
    Segments a grayscale semantic mask and converts it into a colored mask.
    Returns: A NumPy array representing the colored mask.
    """
    # Identify unique grayscale values
    unique_values = sorted(list(set(gray_mask.flatten())))
    
    # Merge values from 238 to 255 into a single segment
    unique_values = [value for value in unique_values if background_threshold < value] + [0]
    
    # Generate a new colormap for the updated unique values
    colormap = plt.get_cmap('jet', len(unique_values))
    colors = (colormap(np.arange(len(unique_values))) * 255).astype(np.uint8)[:, :3]
    
    # Generate the updated colored segmentation mask
    colored_mask = np.zeros((gray_mask.shape[0], gray_mask.shape[1], 3), dtype=np.uint8)
    for i, value in enumerate(unique_values):
        if value == 0:
            colored_mask[gray_mask < background_threshold] = colors[i]
        else:
            colored_mask[gray_mask == value] = colors[i]
    return colored_mask


# Cut and paste avatar from original img

def binary_mask(gray_mask, background_threshold = 140):
    # Apply Otsu's thresholding
    _, binary_mask = cv2.threshold(gray_mask, background_threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_mask, 255 - binary_mask
    
    
# Function to combine a region from 'generated_img' with the rest of 'source_img'

def cut_paste(source_img, generated_img, mask, inv_mask):
    # Repeat mask for each channel if the source and generated images are in color
    if len(source_img.shape) == 3:
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        inv_mask = np.repeat(inv_mask[:, :, np.newaxis], 3, axis=2)

    # Cut out the part of the source image where the inverse mask is non-zero.
    # The inverse mask 'inv_mask' defines the area to keep from the source image.
    avatar_cut = cv2.bitwise_and(source_img, inv_mask)
    
    # Cut out the part of the generated image where the mask is non-zero.
    # The mask 'mask' defines the area to keep from the generated image.
    generated_black_avatar = cv2.bitwise_and(generated_img, mask)
    
    # Add the two images together. This overlays the cut out of 'generated_img'
    # on top of the area that was removed from 'source_img'.
    # The result is a combination of both images in the specified mask area.
    result = cv2.add(avatar_cut, generated_black_avatar)
    
    # Return the final image which has the region from 'generated_img' pasted
    # onto the 'source_img'.
    return result


def avatarcutpaster(input_folder_path, out_folder_path, output_img_size, timestamp):
    input_folder_path = Path(input_folder_path)
    out_folder_path = Path(out_folder_path)
    gen_image_path = out_folder_path / 'generated' / Path(input_folder_path.name + timestamp)

    img_list = list(input_folder_path.glob('*.jpg'))
    generated_img_list = list(gen_image_path.glob('*.jpg'))

    if len(img_list) != len(generated_img_list):
        print('Error: Input and output image list are not matching in length.')

    for img_path in img_list:
        # gen_img_path = input_folder_path / img_path.name
        mask_name = img_path.name.replace('test', 'Image')
        mask_path = input_folder_path / 'Semantic Segmentation' / mask_name
        
        # Load images
        image = Image.open(img_path).resize(output_img_size, Image.ANTIALIAS)
        gen_image = Image.open(gen_image_path / img_path.name).resize(output_img_size, Image.ANTIALIAS)
        mask_image = Image.open(mask_path)
        mask = np.array(mask_image)
        if mask.ndim == 3: mask = mask[:, :, 0]
        inv_mask, mask = binary_mask(mask)
        
        # Convert PIL Images to NumPy arrays before processing
        image_np = np.array(image)
        gen_image_np = np.array(gen_image)
        mask_image_np = np.array(Image.fromarray(mask).resize(output_img_size, Image.ANTIALIAS))
        inv_mask_image_np = np.array(Image.fromarray(inv_mask).resize(output_img_size, Image.ANTIALIAS))

        # Assuming cut_paste is a function that combines images based on the masks
        cut_paste_result = cut_paste(image_np, gen_image_np, mask_image_np, inv_mask_image_np)
        
        # Convert the result back to a PIL Image to display/save
        cut_paste_result_img = Image.fromarray(cut_paste_result)

        # Define the full path including the timestamped folder
        cut_pasted_out_folder = out_folder_path / 'cutpasted' / (input_folder_path.name + timestamp)
        cut_pasted_out_folder.mkdir(parents=True, exist_ok=True)

        # Define the path for the output image within this new directory
        out_img_path = cut_pasted_out_folder / img_path.name
        cut_paste_result_img.save(out_img_path)