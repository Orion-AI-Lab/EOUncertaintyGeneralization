import torch
import torch.utils
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import EuroSAT, Sen12MS, Woody, BigEarthNet, Waititu, FLAIR, Marida, TreeSat, MLRSNet
from .constants import * 
from .uncertainty_utils import create_model
from .webdataset_loaders import create_webdataset_loaders
import pyjson5 as json
import os
import wandb
from torch.utils.data import TensorDataset
import warnings
import numpy as np
import math

def init_wandb(configs):
    """
    Initialize wandb if specified in the configurations.

    Args:
        configs (dict): Run configurations
    """
    if configs['wandb']:
        wandb.init(project=configs['wandb_project'], entity=configs['wandb_entity'], config=configs)
    else:
        return


def load_configs(configs_path):
    """
    Load and combine all configurations from the specified directory.

    Args:
        configs_path (str): Root directory of the configuration files.

    Returns:
        dict: configurations
    """
    main_config_path = os.path.join(configs_path, "configs.json")
    data_config_path = os.path.join(configs_path,"data" ,"data_configs.json")
    stats_config_path = os.path.join(configs_path,"stats" ,"stats.json")
    inference_config_path = os.path.join(configs_path,"inference" ,"inference_configs.json")
    train_config_path = os.path.join(configs_path,"train" ,"train_configs.json")
    #Load configurations
    main_configs=json.load(open(main_config_path))
    data_configs=json.load(open(data_config_path))
    
    if data_configs['webdataset']:
        webdataset_config_path = os.path.join(configs_path,"data" ,"webdataset_configs.json")
        webdataset_configs = json.load(open(webdataset_config_path))
    else:
        webdataset_configs = {}
    stats_configs=json.load(open(stats_config_path))
    if main_configs['task'] == "inference" or main_configs['task'] == "prediction_uncertainty":
        inference_configs=json.load(open(inference_config_path))
        datasets = [main_configs['dataset']]
    else:
        inference_configs = {}
        
    if main_configs['task'] == "train_uncertainties" or main_configs['task'] == "save_features":
        train_configs=json.load(open(train_config_path))
        datasets = train_configs['test_datasets'] + [main_configs['dataset']]
    else:
        train_configs = {}
    
    if main_configs['task'] == "wds_write_parallel":
        datasets = [main_configs['dataset']]
    
    stats = {}
    if 'pretraining_dataset' in inference_configs and inference_configs['pretraining_dataset'] != "imagenet":
        stats[inference_configs['pretraining_dataset'] + '_stats']=stats_configs[inference_configs['pretraining_dataset']]
    else:
        stats['imagenet_stats'] = {"mean": IMAGENET_DEFAULT_MEAN, "std": IMAGENET_DEFAULT_STD}

    print(train_configs)

    for dataset in datasets:
        stats[dataset + '_stats'] = {}
        stats[dataset + '_stats']['mean'] = stats_configs[dataset]['mean']
        stats[dataset + '_stats']['std'] = stats_configs[dataset]['std']
    
    #Merge configurations
    configs = {**main_configs, **data_configs, **stats, **inference_configs, **train_configs, **webdataset_configs}
    print(configs)

    if configs['webdataset'] and "feature_loader" in configs and configs['feature_loader']:
        warnings.warn("Feature loader is not supported with webdataset. Setting webdataset to False.")
        configs['webdataset'] = False
    return configs
    
def get_pretrained_model(configs):
    """
    Retrieve a pretrained uncertainty model as published at 
    https://github.com/mkirchhof/url.

    This function loads a pretrained uncertainty model based on the provided configurations (model name, device).
    
    Parameters:
        configs (dict): A dictionary containing run configurations, including model name 
            and other settings.

    Returns:
        torch.nn.Module: A pretrained uncertainty model ready for use.

    Example:
        >>> model = get_pretrained_model({'model_name': 'resnet50', 'device': 'cuda'})
        >>> print(model)
    """
    pretrained_model = configs['pretraining_dataset']
    if pretrained_model == 'imagenet':
        pretrained_configs = PRETRAINED_IMAGENET_CONFIGS
    elif pretrained_model == 'bigearthnet':
        pretrained_configs = PRETRAINED_BIGEARTHNET_CONFIGS
    elif pretrained_model == 'sen12ms':
        pretrained_configs = PRETRAINED_SEN12MS_CONFIGS
    elif pretrained_model == 'bigearthnet5':
        pretrained_configs = PRETRAINED_BIGEARTHNET5_CONFIGS
    elif pretrained_model == 'bigearthnet_sar':
        pretrained_configs = PRETRAINED_BIGEARTHNET_SAR_CONFIGS
    elif pretrained_model == 'bigearthnet_ms':  
        pretrained_configs = PRETRAINED_BIGEARTHNET_MS_CONFIGS
    elif pretrained_model == 'bigearthnet19_16':
        pretrained_configs = PRETRAINED_BIGEARTHNET19_16_CONFIGS
    elif pretrained_model == 'bigearthnet19_30':
        pretrained_configs = PRETRAINED_BIGEARTHNET19_30_CONFIGS
    elif pretrained_model == 'bigearthnet19_60':
        pretrained_configs = PRETRAINED_BIGEARTHNET19_60_CONFIGS
    elif pretrained_model == 'flair':
        pretrained_configs = PRETRAINED_FLAIR_CONFIGS
    else:
        raise NotImplementedError(f"Dataset {pretrained_model} with uncertainty module has not been trained.")
    
    vit_variant = configs['model'] 

    args = pretrained_configs[vit_variant]

    # vit_variant = configs['model']   
    # unc_width=pretrained_configs[vit_variant]["unc_width"]
    # unc_depth=pretrained_configs[vit_variant]["unc_depth"]
    # model=pretrained_configs[vit_variant]["name"]
    # unc_module=pretrained_configs[vit_variant]["uncertainty_module"]
    # stopgrad=pretrained_configs[vit_variant]["stopgrad"]
    # checkpoint_path = pretrained_configs[vit_variant]["uncertainty_checkpoint"]#configs['uncertainty_checkpoint']

    model = create_model(**args)

    # model = create_model(model, unc_module=unc_module, unc_width=unc_width, unc_depth=unc_depth, pretrained=True, uncertainty_checkpoint=checkpoint_path, stopgrad=stopgrad)
    model.to(configs['device'])
    return model


def get_dataset(configs,dataset, mode, download=False,dataset_root=None,imagenet_norm=False):
    """_summary_

    Args:
        configs (_type_): _description_
        mode (_type_): _description_
    """
    if dataset.lower() == "eurosat":
        if imagenet_norm:
            transform = transforms.Compose([transforms.Resize((224, 224)),transforms.Normalize(mean=configs[configs['pretraining_dataset']+"_stats"]['mean'], std=configs[configs['pretraining_dataset']+"_stats"]['std'])])
        else:
            transform = None
        data =EuroSAT.EuroSAT(root=dataset_root, bands=EuroSAT.EuroSAT.BAND_SETS['rgb'],transforms=transform,download=download,split=mode)
    elif dataset.lower() == "sen12ms":
        data = Sen12MS.Sen12MSDataset(configs, dataset, mode)
    elif dataset.lower() == "woody":
        data = Woody.WoodyDataset(configs, dataset, mode)
    elif dataset.lower() == "bigearthnet":
        data = BigEarthNet.BigEarthNetDataset(configs, dataset, mode)
    elif dataset.lower() == "waititu":
        data = Waititu.WaitituDataset(configs, dataset, mode)
    elif dataset.lower() == "flair":
        data = FLAIR.FLAIRDataset(configs, dataset, mode)
    elif dataset.lower() == "marida":
        data = Marida.GenDEBRIS(configs, dataset, mode) #mode=mode, path=dataset_root, mode = 'train', transform=None, standardization=None, path = "/mnt/nvme1/nbountos/MARIDA", agg_to_water= True)
    elif dataset.lower() == "treesat":
        data = TreeSat.TreeSatAIDataset(configs, mode)
    elif dataset.lower() == "mlrsnet":
        data = MLRSNet.MLRSNet(configs, mode)
    else:
        raise NotImplementedError(f"Dataset {dataset} is not implemented.")
    return data

def check_path(path):
    if not os.path.exists(path):
        print(f"Path {path} does not exist.")
        return False
    return True

def get_loaders(dataset, download, dataset_root, features_root=None, batch_size=1, num_workers=1, pin_memory=False,imagenet_norm=False, feature_load=False, configs=None):
    """
    Get data loaders for the specified dataset.

    This function creates and returns PyTorch data loaders for training, validation, and testing 
    of a given dataset.
    
    Parameters:
        dataset (str): The name of the dataset.
        download (bool): Whether to download the dataset if it is not already available.
        dataset_root (str): The root directory where the dataset is stored or will be downloaded.
        features_root (str): Directory load extracted features for the dataset.
        batch_size (int): The number of samples per batch to load.
        num_workers (int): The number of subprocesses to use for data loading.
        pin_memory (bool): If True, data loaders will copy Tensors into CUDA pinned memory for 
            faster data transfer to GPU.
        imagenet_norm (bool, optional): Whether to apply ImageNet normalization. This is typically 
            used when evaluating models pretrained on ImageNet. Defaults to False.
        feature_load (bool, optional): If True, loads pre-extracted features instead of raw data. 
            Defaults to False.
        configs (dict): A dictionary containing additional run configurations.

    Raises:
        NotImplementedError: Raised if the specified dataset is not supported.

    Returns:
        tuple: A tuple containing data loaders for training, validation, and test sets.
    """
    '''TODO: Add option to choose between RGB and multispectral bands for EuroSAT/Sen12MS dataset'''
    if not configs['webdataset']:
        if features_root is not None and feature_load:
            model_name = "_".join(configs['model_name'].split("_")[:2])
            features_path = os.path.join(features_root, model_name, dataset)
            train_path = os.path.join(features_path, 'train')
            val_path = os.path.join(features_path, 'val')
            test_path = os.path.join(features_path, 'test')
            
            #Check if all_features.npy are available for train val test
            if not check_path(train_path):
                raise FileNotFoundError(f"Path {train_path} does not exist.")
            if not check_path(val_path):
                raise FileNotFoundError(f"Path {val_path} does not exist.")
            if not check_path(test_path):
                raise FileNotFoundError(f"Path {test_path} does not exist.")
            
            #Load npy features
            train_features = torch.from_numpy(np.load(os.path.join(train_path, 'all_features.npy')))
            val_features = torch.from_numpy(np.load(os.path.join(val_path, 'all_features.npy')))
            test_features = torch.from_numpy(np.load(os.path.join(test_path, 'all_features.npy')))

            #Load labels
            train_labels = torch.from_numpy(np.load(os.path.join(train_path, 'all_targets.npy')))
            val_labels = torch.from_numpy(np.load(os.path.join(val_path, 'all_targets.npy')))
            test_labels = torch.from_numpy(np.load(os.path.join(test_path, 'all_targets.npy')))
            
            if configs["unc_module"] == "pred-transformer":
                train_tokens = torch.from_numpy(np.load(os.path.join(train_path, 'all_tokens.npy')))
                val_tokens = torch.from_numpy(np.load(os.path.join(val_path, 'all_tokens.npy')))
                test_tokens = torch.from_numpy(np.load(os.path.join(test_path, 'all_tokens.npy')))
        
                train_data = TensorDataset(train_features, train_tokens, train_labels)
                val_data = TensorDataset(val_features, val_tokens, val_labels)
                test_data = TensorDataset(test_features, test_tokens, test_labels)
            else:
                train_data = TensorDataset(train_features, train_labels)
                val_data = TensorDataset(val_features, val_labels)
                test_data = TensorDataset(test_features, test_labels)
        else:
            train_data = get_dataset(configs,dataset,mode="train",download=download,dataset_root=dataset_root,imagenet_norm=imagenet_norm)
            val_data = get_dataset(configs,dataset, mode="val", download=download,dataset_root=dataset_root,imagenet_norm=imagenet_norm)
            test_data = get_dataset(configs,dataset,mode="test",download=download,dataset_root=dataset_root,imagenet_norm=imagenet_norm)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=configs['shuffle'], num_workers=num_workers, pin_memory=pin_memory)
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    else:
        train_loader, val_loader, test_loader = create_webdataset_loaders(configs)
    return train_loader, val_loader, test_loader


def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"]
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - configs["warmup_epochs"]) / (configs["epochs"] - configs["warmup_epochs"]))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


'''_,_, test_loader = get_loaders("eurosat", True, "/home/mila/n/nikolaos.bountos/scratch/EvalDatasets/", 32, 4, True)

sample, label = next(iter(test_loader))
print(sample.shape, label.shape)
print(sample.mean())
#Print imagenet mean and std
print(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)'''