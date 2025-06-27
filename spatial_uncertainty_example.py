import torch.nn as nn
from utilities import utils, uncertainty_utils
import torch
import timm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from torch.nn import functional as F
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

"""
    Example for localized spatial uncertainty estimation.
"""

def plot_image_unc(image, patches, total_uncertainty,dataset,save_dir,it,mean,std):
    """
    Plot image and its patches resampled to image size as called in infer()
    """
    # Interpolate patches to match the image size
    print(f"Image shape: {image.shape}")
    print(f"Patches shape: {patches.shape}")
    patches = F.interpolate(patches.unsqueeze(0).unsqueeze(0), size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False)

    # Convert patches to numpy for plotting
    patches = patches.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    image = image.permute(1, 2, 0).detach().cpu().numpy()  # Ensure image is in (H, W, C)

    # Plot image and patches side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    #denormalize image using mean and std  
    image = image*std + mean
    ax[0].imshow(image/image.max())
    ax[0].set_title("Image")
    ax[0].axis('off')
    print(f"Patches min: {patches.min()}, max: {patches.max()}")
    
    im = ax[1].imshow(patches)
    ax[1].set_title("Patches (Resampled)")
    ax[1].axis('off')
    cbar = fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    cbar.set_label("Patch Uncertainty")
    fig.suptitle("Total Uncertainty: " + str(total_uncertainty.item()))
    plt.tight_layout()
    #make dir
    os.makedirs(save_dir+'/'+dataset, exist_ok=True)
    plt.savefig(save_dir+'/'+dataset+ '/'+dataset+"_image_patches_"+str(it)+".png")
    plt.close('all')

    
def infer(model, test_loader,dataset,mean,std):
    model.eval()
    with torch.no_grad():
        for it, b in enumerate(test_loader):
            input, target = b
            total, token = model(input, patches=True)
            token = token.squeeze(2)
            #patches per row
            patches_per_row = int(np.sqrt(token.shape[1]))
            patches = token.reshape(-1,patches_per_row,patches_per_row)
            
            _, total_uncertainty,_ = total
            #Get index with highest uncertainty

            index = torch.argmax(total_uncertainty)
            print(f"Index with highest uncertainty: {index}")
            print(f"Total uncertainty at index: {total_uncertainty[index]}")
            plot_image_unc(input[index,:,:], patches[index],total_uncertainty[index],dataset,save_dir="unc_plots/low_uncertainties/",it=it,mean=mean,std=std)
            if it > 20:
                return
        
configs = {
        "pretraining_dataset":"flair",
        "model":"vit-tiny",
        "device":"cpu",
}

model = utils.get_pretrained_model(configs)
model.eval()

configs = utils.load_configs("configs")
mean = list(reversed(configs['marida_stats']["mean"][:3]))
std = list(reversed(configs['marida_stats']["std"][:3]))

_,_, test_loader = utils.get_loaders(configs['dataset'], download=True, dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)
infer(model,test_loader,configs['dataset'],mean,std)