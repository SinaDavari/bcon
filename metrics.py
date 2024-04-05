# Determining the Frechet Inception Distance (FID) and Structural Similarity Index (SSIM) Metrics
# pip install torchvision scikit-image

import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import os
from scipy.linalg import sqrtm

def load_images(path, resize=None):
    images = []
    for filename in sorted(os.listdir(path)):
        if filename.endswith('.jpg'):
            img = Image.open(os.path.join(path, filename)).convert('RGB')
            if resize:
                img = img.resize(resize, Image.ANTIALIAS)
            images.append(img)
    return images

def calculate_fid(act1, act2):
    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)
    ssdiff = np.sum((mu1 - mu2) ** 2)
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_activations(images, model, device):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    activations = []
    for img in images:
        input_tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(input_tensor)
        activations.append(pred.cpu().numpy().squeeze())
    return np.array(activations)

def compute_fid_and_ssim(path1, path2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    model.eval()

    img_size = 1280
    images1 = load_images(path1, resize=(img_size, img_size))
    images2 = load_images(path2)

    act1 = compute_activations(images1, model, device)
    act2 = compute_activations(images2, model, device)

    fid_value = calculate_fid(act1, act2)

    ssim_values = []
    for img1, img2 in zip(images1, images2):
        img1 = np.array(img1)
        img2 = np.array(img2)
        ssim_value = ssim(img1, img2, multichannel=True)
        ssim_values.append(ssim_value)
    return fid_value, np.mean(ssim_values)
    
    
# Paths to your datasets
# BlendCon output
path_to_synthetic = "/home/sina/Downloads/controlnet_p/ValueOfRealism/Synthetic Dataset/Sample 6"

# BCGen output
path_to_generated = "/home/sina/Downloads/controlnet_p/ValueOfRealism/Generated Images/Sample 6"

# A Real Image
path_to_real = "/home/sina/Downloads/testbcgenreal"

fid, average_ssim = compute_fid_and_ssim(path_to_synthetic, path_to_real)
print(f"FID Score: {fid: .2f}")
print(f"Average SSIM: {average_ssim: .2f}")