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
import kornia
from timm.models._helpers import load_checkpoint
import segmentation_models_pytorch as smp    
import heapq
import pyjson5 as json
import cv2

"""
Example for localized uncertainty estimation,
"""

dataset_capitalized = {"flair":"Flair","waititu":"Waititu","woody":"Woody","marida":"Marida", "bigearthnet":"BigEarthNet"}

def plot_image_unc(image, patches, total_uncertainty, dataset, save_dir, it, mean, std, cls_attention=None, loss_plot=None): 
    """
    Plot image and its patches resampled to image size as called in infer()
    """
    patches = F.interpolate(patches.unsqueeze(0).unsqueeze(0), size=(image.shape[1], image.shape[2]), mode='bilinear', align_corners=False)
    patches = patches.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    image = kornia.enhance.denormalize(image.unsqueeze(0), mean, std)
    image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()

    # Determine number of subplots dynamically
    subplots = [image / image.max(), patches]  
    titles = ["Image", "Uncertainty"]  
    
    if cls_attention is not None:
        #Option to plot cls token attention
        subplots.append(cls_attention.squeeze().detach().cpu().numpy())
        titles.append("Cls Attention")
    
    if loss_plot is not None:
        subplots.append(loss_plot.squeeze().detach().cpu().numpy())
        titles.append("Downstream Loss")

    num_plots = len(subplots)
    
    # Create figure and axes with tight layout
    fig, ax = plt.subplots(1, num_plots, figsize=(4 * num_plots, 4), gridspec_kw={'wspace': 0.05}, constrained_layout=True)

    # Plot each subplot
    for i, (data, title) in enumerate(zip(subplots, titles)):
        if i == 0:
            im = ax[i].imshow(data/data.max(), cmap='viridis')
            vmin = (data/data.max()).min()
        else:
            ax[i].imshow(data, cmap='viridis')
        ax[i].set_title(title,fontsize=24)
        ax[i].axis('off')

    # Add a single colorbar below the figure
    cbar_ax = fig.add_axes([1.01, 0.01, 0.015, 0.9])  # Adjust the position and size
    cbar = fig.colorbar(im, cax=cbar_ax, orientation="vertical", pad=0.02)
    

    vmax = 1.
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels(["Low", "High"],fontsize=24)
    # Save figure
    os.makedirs(f"{save_dir}/{dataset}", exist_ok=True)
    plt.savefig(f"{save_dir}/{dataset}/{dataset}_image_patches_{it}_uncertainty_{round(total_uncertainty,2)}.png", bbox_inches='tight', pad_inches=0.0,dpi=300)
    plt.close()

def plot_uncertain(image,dataset,save_dir,it,mean,std):
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    image = kornia.enhance.denormalize(image.unsqueeze(0), mean, std)
    image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
    ax.imshow(image/image.max())
    ax.axis('off')
    #Add title to ax
    ax.set_title(dataset_capitalized[dataset],fontsize=24)
    os.makedirs(save_dir+'/'+dataset, exist_ok=True)
    plt.savefig(save_dir+'/'+dataset+ '/'+dataset+"_image_"+str(it)+".png", bbox_inches='tight', pad_inches=0.1)
    plt.close('all')


def infer(model, test_loader,dataset,mean,std,supervised_model=None,module='pred-net',plot_uncertainties=None,save_dir="uncertainties_top_plots/",save_image=False):
    model.eval()
    max_unc =0.0
    min_unc = 10000.
    device = 'cuda:1'
    #Keep top 20 images with highest/lowest uncertainties
    max_heap_least_uncertainty = [] 
    min_heap_highest_uncertainty = []  
    with torch.no_grad():
        max_uncertainty = 0
        min_uncertainty = 1000000.
        for it, b in enumerate(test_loader):
            input, target = b
            input = input.to(device)
            target = target.to(device)
            total, token = model(input, patches=True)
            
            if supervised_model is not None:
                #Calculate loss for visualization
                supervised_model.to(device)
                pred = supervised_model(input)
                loss = F.cross_entropy(pred,target, reduction='none').cpu()
            token = token.cpu()
            input = input.cpu().detach()
            target = target.cpu().detach()
            
            if module=="pred-transformer" and model.unc_module.pool == 'cls':
                token = token[1:,]
                cls_token = token[0,:]
            token = token.squeeze(2)
            patches_per_row = int(np.sqrt(token.shape[1]))
            patches = token.reshape(-1,patches_per_row,patches_per_row)
            
            _, total_uncertainty,_ = total
            total_uncertainty = total_uncertainty.cpu()
            
            #Get index with highest uncertainty

            index = torch.argmax(total_uncertainty)
            index_min = torch.argmin(total_uncertainty)

            
            if len(total_uncertainty.shape)==0:
                total_uncertainty = total_uncertainty.unsqueeze(0)
            if total_uncertainty[index_min] < min_uncertainty:
                min_uncertainty = total_uncertainty[index_min]
                image_to_plot_min = input[index_min,:,:]
                
            if supervised_model is not None:
                loss_plot = loss[index]
            else:
                loss_plot = None
            
            heapq.heappush(min_heap_highest_uncertainty, (total_uncertainty[index].item(), it,input[index,:,:].cpu().numpy()))   
            if len(min_heap_highest_uncertainty) > 20:
                heapq.heappop(min_heap_highest_uncertainty)
            
            heapq.heappush(max_heap_least_uncertainty, (-total_uncertainty[index_min].item(),it, input[index_min.item(), :, :].cpu().numpy()))
            if len(max_heap_least_uncertainty) > 400:
                heapq.heappop(max_heap_least_uncertainty) 
                
            total_uncertainty = total_uncertainty[index].item()
            
            if plot_uncertainties is not None:
                #Plot Per pixel uncertainty for the image with highest uncertainty in the batch
                plot_image_unc(input[index,:,:], patches[index],total_uncertainty,dataset,save_dir=save_dir,it=it,mean=mean,std=std,loss_plot=loss_plot)
            
    if save_image:
        for it, (unc,_,image) in enumerate(min_heap_highest_uncertainty):
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            image = torch.from_numpy(image)
            image = image.squeeze()
            
            image = kornia.enhance.denormalize(image.unsqueeze(0), mean, std)
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            if dataset=='marida' or dataset=='mlrsnet' or dataset=='bigearthnet':
                image = image/image.max()
                image = image*255
            #save image as png using cv2
            os.makedirs(f"{save_dir}", exist_ok=True)
            os.makedirs(f"{save_dir}/high_uncertainty", exist_ok=True)
            os.makedirs(f"{save_dir}/high_uncertainty/{dataset}", exist_ok=True)

            image = cv2.cvtColor((image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/high_uncertainty/{dataset}/{dataset}_image_{it}_uncertainty_{round(unc,2)}.png",image)
        for it, (unc,_,image) in enumerate(max_heap_least_uncertainty):
            mean = torch.tensor(mean)
            std = torch.tensor(std)
            image = torch.from_numpy(image)

            image = image.squeeze()
            image = kornia.enhance.denormalize(image.unsqueeze(0), mean, std)
            image = image.squeeze().permute(1, 2, 0).detach().cpu().numpy()
            if dataset=='marida' or dataset=='mlrsnet' or dataset=='bigearthnet':
                image = image/image.max()
                image = image*255
            #save image as png using cv2
            os.makedirs(f"{save_dir}", exist_ok=True)
            os.makedirs(f"{save_dir}/low_uncertainty/{dataset}", exist_ok=True)
            image = cv2.cvtColor((image), cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{save_dir}/low_uncertainty/{dataset}/{dataset}_image_{it}_uncertainty_{round(-unc,2)}.png",image)
    
    print(f"Max uncertainty: {max_unc}, Min uncertainty: {min_unc}")       

def construct_model(module='pred-transformer',pool="mean",train_path= "vit_tiny/flair/best_model.pth",adapt_patch_size=None):
    configs={}
    configs['unc_module'] = module
    configs['model_name'] = 'vit_large_patch16_224'
    configs['in_channels'] =3
    configs['image_size'] = 512
    configs['pretrained_classes'] = 19
    configs['trained_model_path'] =train_path
    configs["unc_width"]= 512
    configs["unc_depth"]= 2
    configs['num_heads']=1
    configs['init_prednet_zero'] = False
    configs['stopgrad'] = True
    configs['pool'] = pool
    configs['task'] = 'plot'
    configs['pretrained_image_size'] = 120
    model = timm.create_model(configs['model_name'],pretrained=False,in_chans=configs["in_channels"], img_size=configs["pretrained_image_size"], num_classes=configs['pretrained_classes'],dynamic_img_pad=True, dynamic_img_size=True)
    model = uncertainty_utils.add_unc_module(model, configs['unc_module'], configs['unc_width'], configs['unc_depth'], configs['init_prednet_zero'],configs['stopgrad'],pool=configs['pool'])
    load_checkpoint(model, configs['trained_model_path'])
    if adapt_patch_size is not None:
        #Adapt patch size if needed
        model.model.patch_embed.set_input_size(configs['image_size'],patch_size=adapt_patch_size)
    model.eval()
    model.cuda()
    return model


def get_supervised_model(in_channels,num_classes,weight_path):
    task_predictor = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=num_classes,  # model output channels (number of classes in your dataset)
        )
    task_predictor.load_state_dict(torch.load(weight_path))
    task_predictor.eval()
    return task_predictor


if __name__=="__main__":
    module="pred-net"
    model = construct_model(pool='mean',module=module,train_path="trained_models/vit_large/bigearthnet/best_model.pth")
    configs = utils.load_configs("configs")
    mean = list(reversed(configs[configs['dataset']+'_stats']["mean"][:3]))
    std = list(reversed(configs[configs['dataset']+'_stats']["std"][:3]))

    _,_, test_loader = utils.get_loaders(configs['dataset'], download=True, dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)
    predictors = {
        "woody":"woody/unet/resnet50/best_model_state_dict_19.pt",
        "flair":"flair/unet/resnet50/best_model_state_dict_19.pt",
        "waititu":"waititu/unet/resnet50/best_model_state_dict_19.pt",
        "marida":"marida/unet/resnet50/best_model_state_dict_19.pt"
    }
    
    # For supervised model None or path to the model + channels, classes
    supervised_model = None 
    stats = json.load(open("configs/stats/stats.json"))
    images = []
    for dataset in ["marida"]:
        print(f"Dataset: {dataset}")
        configs['dataset'] = dataset
        _,_, test_loader = utils.get_loaders(configs['dataset'], download=True, dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)

        if dataset=="bigearthnet":
            mean = list(reversed(stats[dataset]["mean"][3:6]))
            std = list(reversed(stats[dataset]["std"][3:6]))
        else:
            print(stats[dataset]["mean"][:3])
            mean = list(reversed(stats[dataset]["mean"][:3]))
            std = list(reversed(stats[dataset]["std"][:3]))
        _ = infer(model,test_loader,dataset,mean,std,supervised_model=supervised_model,module=module,save_dir="UncertaintyVisualizations/",save_image=False,plot_uncertainties=True)