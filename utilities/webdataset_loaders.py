import torch
import random
import os
import io
import webdataset as wds
import einops
from torchvision import transforms
import numpy as np
import glob
import kornia
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import albumentations as A

def create_webdataset_loaders(
    configs, repeat=False):
    """Create webdataset loaders for training, validation and testing
       Assumes the webdasatet exists either in root/webdataset/dataset_name or in webdataset_root_path/webdataset/dataset_name

    Args:
        configs (_type_): configuration dictionary
        repeat (bool, optional): Defaults to False.
    """
    
    def get_patches(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"]),weights_only=True).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]),weights_only=True)  
                   
            if configs['dataset']=='sen12ms':
                label[label>=configs['threshold']] = 1
                label[label<configs['threshold']] = 0
            if configs["normalization"] == "minmax":
                if configs["dataset"] == "sen12ms":
                    s1 = image[:2, :, :] / 25 + 1
                    s2 = image[2:, :, :] / 10000
                    image = torch.cat((s1, s2), dim=0)
                else:
                    image /= image.max() + 1e-6
            elif configs["normalization"] == "standard":
                if (configs["webdataset"] and (configs['task']=="inference" or configs['task']=="save_features" or configs["task"]=="prediction_uncertainty" or configs["task"]=="train_uncertainties")) or ("eval_webdataset" in configs and configs["eval_webdataset"]):
                    configs["mean"] = configs[configs["dataset"]+"_stats"]["mean"]
                    configs["std"] = configs[configs["dataset"]+"_stats"]["std"]
                if "mean" not in configs or "std" not in configs:
                    print("Mean and Std not provided for this dataset. Exiting!")
                    exit(2)
                                
                if configs['dataset']=='mlrsnet':
                    mean = configs["mean"]
                    std = configs["std"]
                    normalization = transforms.Normalize(mean=mean, std=std)
                    if configs['pretraining_dataset']=="imagenet":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                if configs['spectral_type'] == 'rgb':
                    if configs['dataset'] == 'marida':
                        b = image[:,:,0]
                        g = image[:,:,1]
                        r = image[:,:,2]
                        image = torch.stack((r,g,b), dim=2)
                        image = einops.rearrange(image, "h w c -> c h w")

                        mean = list(reversed(configs["mean"][:3]))
                        std = list(reversed(configs["std"][:3]))
                        if configs['recall_per_dataset']['marida'] == 'multilabel':
                            label[label==-1] = 11
                            num_classes = configs['num_classes']
                            unique_labels = torch.unique(label)
                            multilabel_target = torch.zeros(num_classes, dtype=torch.float)
                            multilabel_target[unique_labels] = 1
                            label = multilabel_target
                        if configs['pretraining_dataset']=="imagenet":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    elif configs["dataset"]=="bigearthnet":
                        b = image[3,:,:]
                        g = image[4,:,:]
                        r = image[5,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        if configs['pretraining_dataset']=="imagenet" and configs['task_image_size']==224 and configs["task"]=="prediction_uncertainty":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                        if configs["image_size"]!= 120:
                            image = image.permute(1, 2, 0).numpy()
                            aug = A.Compose(
                                [
                                    A.augmentations.Resize(
                                        height=configs["image_size"],
                                        width=configs["image_size"],
                                        p=1.0,
                                    )
                                ]
                            )(image=image)
                            image = aug["image"]
                            image = einops.rearrange(image, "h w c -> c h w")
                            image = torch.from_numpy(image).float()
                    elif configs['dataset'] == "treesat_aerial":
                        image = image[1:,:,:]
                        mean = configs["mean"][1:]
                        std = configs["std"][1:]
                    elif configs["dataset"] == "treesat_sen200m":
                        b = image[0,:,:]
                        g = image[1,:,:]
                        r = image[2,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][:3]))
                        std = list(reversed(configs["std"][:3]))
                    elif configs["dataset"] == "waititu":
                        image = image[:3,:,:]
                        mean = configs["mean"]
                        std = configs["std"]
                    elif configs["dataset"] == "woody":
                        image = image[:3,:,:]
                        mean = configs["mean"]
                        std = configs["std"]
                    elif configs["dataset"] == "sen12ms":
                        r = image[5,:,:]
                        g = image[4,:,:]
                        b = image[3,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        if configs['pretraining_dataset']=="imagenet":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    elif configs["dataset"] == "flair":
                        image = image[:3,:,:]
                        mean = configs["mean"][:3]
                        std = configs["std"][:3]
                        if configs['multilabel']:
                            num_classes = configs['num_classes']
                            unique_labels = torch.unique(label)
                            multilabel_target = torch.zeros(num_classes, dtype=torch.float)
                            multilabel_target[unique_labels] = 1
                            label = multilabel_target
                        if configs['pretraining_dataset']=="imagenet":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    normalization = transforms.Normalize(mean=mean, std=std)
                elif configs['spectral_type'] == 'multispectral':
                    if configs["dataset"]=="bigearthnet":
                        image = image[2:,:,:]
                        mean = configs["mean"][2:]
                        std = configs["std"][2:]
                    normalization = transforms.Normalize(mean=mean, std=std)
                elif configs['spectral_type'] == 'sar':
                    if configs["dataset"]=="bigearthnet":
                        image = image[:2,:,:]
                        mean = configs["mean"][:2]
                        std = configs["std"][:2]
                    normalization = transforms.Normalize(mean=mean, std=std)
                else:
                    normalization = transforms.Normalize(mean=configs["mean"], std=configs["std"])
                image = normalization(image)
            elif configs["normalization"] == "imagenet" and configs['dataset']=='bigearthnet':
                b = image[2,:,:]
                g = image[3,:,:]
                r = image[4,:,:]
                image = torch.stack((r,g,b), dim=0)
                image /= image.max()
                resize = transforms.Compose([transforms.Resize((224, 224))])
                image = resize(image)
                normalization = transforms.Normalize(mean=configs[configs['pretraining_dataset']+"_stats"]['mean'], std=configs[configs['pretraining_dataset']+"_stats"]['std'])
                image = normalization(image)
            
            if configs["task"] == "train_uncertainties" or configs["task"] == "prediction_uncertainty" or configs["task"] == "save_features":
                if configs['dataset']=='bigearthnet' and configs['pretrained_classes']==5:
                    inverted_nomenclature_five = {2: 0, 14: 0, 0: 1, 16: 1, 3: 1, 6: 1, 15: 1, 13: 1, 17: 2, 10: 2, 12: 2, 1: 2, 5: 2, 7: 2, 18: 2, 8: 3, 9: 3, 4: 4, 11: 4}
                    l5 = torch.zeros(5)
                    for i,_ in enumerate(label):
                        l5[inverted_nomenclature_five[i]] = l5[inverted_nomenclature_five[i]] or label[i]
                    label = l5
                
            yield (image, label)

    def get_patches_eval(src):
        for sample in src:
            image = torch.load(io.BytesIO(sample["image.pth"]),weights_only=True).float()
            label = torch.load(io.BytesIO(sample["labels.pth"]),weights_only=True)
            
            if configs['dataset']=='sen12ms':
                label[label>=configs['threshold']] = 1
                label[label<configs['threshold']] = 0
                
            if configs["normalization"] == "minmax":
                if configs["dataset"] == "sen12ms":
                    s1 = image[:2, :, :] / 25 + 1
                    s2 = image[2:, :, :] / 10000
                    image = torch.cat((s1, s2), dim=0)
                else:
                    image /= image.max() + 1e-6
            elif configs["normalization"] == "standard":
                if (configs["webdataset"] and (configs['task']=="inference" or configs['task']=="save_features" or configs["task"]=="prediction_uncertainty" or configs["task"]=="train_uncertainties")) or ("eval_webdataset" in configs and configs["eval_webdataset"]):
                    configs["mean"] = configs[configs["dataset"]+"_stats"]["mean"]
                    configs["std"] = configs[configs["dataset"]+"_stats"]["std"]
                if "mean" not in configs or "std" not in configs:
                    print("Mean and Std not provided for this dataset. Exiting!")
                    exit(2)
                
                if configs['dataset']=='mlrsnet':
                    mean = configs["mean"]
                    std = configs["std"]
                    normalization = transforms.Normalize(mean=mean, std=std)
                    if configs['pretraining_dataset']=="imagenet" and configs['task']!="prediction_uncertainty":
                        #Resize to 224x224
                        resize = transforms.Compose([transforms.Resize((224, 224))])
                        image = resize(image)
                if configs['spectral_type'] == 'rgb':
                    if configs['dataset'] == 'marida':
                        b = image[:,:,0]
                        g = image[:,:,1]
                        r = image[:,:,2]
                        label[label == -1] = 11
                        image = torch.stack((r,g,b), dim=2)
                        image = einops.rearrange(image, "h w c -> c h w")
                        mean = list(reversed(configs["mean"][:3]))
                        std = list(reversed(configs["std"][:3]))
                        if configs['recall_per_dataset']['marida'] == 'multilabel':
                            label[label==-1] = 11
                            num_classes = configs['task_num_classes']
                            unique_labels = torch.unique(label)
                            multilabel_target = torch.zeros(num_classes, dtype=torch.float)
                            multilabel_target[unique_labels] = 1
                            label = multilabel_target
                        if configs['pretraining_dataset']=="imagenet" and configs['task_image_size']==224 and configs['task']=="prediction_uncertainty":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    elif configs["dataset"]=="bigearthnet":
                        b = image[3,:,:]
                        g = image[4,:,:]
                        r = image[5,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        # if not configs['multilabel']:
                        #     label = torch.sum(label, dim=0).long()
                        
                        if configs['pretraining_dataset']=="imagenet": #and configs['task_image_size']==224 and configs['task']=="prediction_uncertainty":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                        if configs["image_size"]!= 120:
                            image = image.permute(1, 2, 0).numpy()
                            aug = A.Compose(
                                [
                                    A.augmentations.Resize(
                                        height=configs["image_size"],
                                        width=configs["image_size"],
                                        p=1.0,
                                    )
                                ]
                            )(image=image)
                            image = aug["image"]
                            image = einops.rearrange(image, "h w c -> c h w")
                            image = torch.from_numpy(image).float()
                    elif configs["dataset"] == "waititu":
                        image = image[:3,:,:]
                        mean = configs["mean"]
                        std = configs["std"]
                    elif configs["dataset"] == "woody":
                        image = image[:3,:,:]
                        mean = configs["mean"]
                        std = configs["std"]
                    elif configs['dataset'] == "treesat_aerial":
                        image = image[1:,:,:]
                        mean = configs["mean"][1:]
                        std = configs["std"][1:]
                    elif configs["dataset"] == "treesat_sen200m":
                        b = image[0,:,:]
                        g = image[1,:,:]
                        r = image[2,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][:3]))
                        std = list(reversed(configs["std"][:3]))
                    elif configs["dataset"] == "sen12ms":
                        r = image[5,:,:]
                        g = image[4,:,:]
                        b = image[3,:,:]
                        image = torch.stack((r,g,b), dim=0)
                        mean = list(reversed(configs["mean"][3:6]))
                        std = list(reversed(configs["std"][3:6]))
                        if configs['pretraining_dataset']=="imagenet":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    elif configs["dataset"] == "flair":
                        image = image[:3,:,:]
                        mean = configs["mean"][:3]
                        std = configs["std"][:3]
                        if configs["image_resize"]:
                            image = image.permute(1, 2, 0).numpy()
                            label = label.numpy()
                            aug = A.Compose(
                                [
                                    A.augmentations.Resize(
                                        height=configs["image_resize"],
                                        width=configs["image_resize"],
                                        p=1.0,
                                    )
                                ]
                            )(image=image, mask=label)
                            image = aug["image"]
                            label = aug["mask"]
                            image = einops.rearrange(image, "h w c -> c h w")
                            image = torch.from_numpy(image).float()
                            label = torch.from_numpy(label).long()
                        if configs['multilabel']:
                            if configs['num_classes']==5:
                                new_target = torch.zeros(5)
                                unique_labels = torch.unique(label)
                                multilabel_target = torch.zeros(19, dtype=torch.float)
                                multilabel_target[unique_labels] = 1
                                reduced_mapping = {0:0,1:0,2:0,12:0,17:0,3:1,4:1,13:1,5:1,6:1,7:1,14:1,15:1,16:1,8:2,10:2,11:2,9:3}
                                for i, new_idx in reduced_mapping.items():
                                    new_target[new_idx] = new_target[new_idx] or multilabel_target[i]
                                label = new_target
                            else:
                                num_classes = configs['num_classes']
                                unique_labels = torch.unique(label)
                                multilabel_target = torch.zeros(num_classes, dtype=torch.float)
                                multilabel_target[unique_labels] = 1
                                label = multilabel_target
                        else:
                            if configs['num_classes']==5:
                                reduced_mapping = {0:0,1:0,2:0,12:0,17:0,3:1,4:1,13:1,5:1,6:1,7:1,14:1,15:1,16:1,8:2,10:2,11:2,9:3}
                                for i, new_idx in reduced_mapping.items():
                                    label[label==i] = new_idx
                                
                        if configs['pretraining_dataset']=="imagenet" and configs['task']!="prediction_uncertainty":
                            #Resize to 224x224
                            resize = transforms.Compose([transforms.Resize((224, 224))])
                            image = resize(image)
                    normalization = transforms.Normalize(mean=mean, std=std)
                elif configs['spectral_type'] == 'multispectral':
                    if configs["dataset"]=="bigearthnet":
                        image = image[2:,:,:]
                        mean = configs["mean"][2:]
                        std = configs["std"][2:]
                    normalization = transforms.Normalize(mean=mean, std=std)
                elif configs['spectral_type'] == 'sar':
                    if configs["dataset"]=="bigearthnet":
                        image = image[:2,:,:]
                        mean = configs["mean"][:2]
                        std = configs["std"][:2]
                    normalization = transforms.Normalize(mean=mean, std=std)
                else:
                    normalization = transforms.Normalize(mean=configs["mean"], std=configs["std"])
                image = normalization(image)
            elif configs["normalization"] == "imagenet":
                if configs['dataset'] == 'bigearthnet':
                    b = image[2,:,:]
                    g = image[3,:,:]
                    r = image[4,:,:]
                    image = torch.stack((r,g,b), dim=0)
                elif configs['dataset'] == 'sen12ms':
                    b = image[3,:,:]
                    g = image[4,:,:]
                    r = image[5,:,:]
                    image = torch.stack((r,g,b), dim=0)
                image /= image.max()
                resize = transforms.Compose([transforms.Resize((224, 224))])
                image = resize(image)
                normalization = transforms.Normalize(mean=configs[configs['pretraining_dataset']+"_stats"]['mean'], std=configs[configs['pretraining_dataset']+"_stats"]['std'])
                image = normalization(image)
            
            if configs["task"] == "train_uncertainties" or configs["task"] == "prediction_uncertainty" or configs["task"] == "save_features":
                if configs['dataset']=='bigearthnet' and configs['pretrained_classes']==5:
                    inverted_nomenclature_five = {2: 0, 14: 0, 0: 1, 16: 1, 3: 1, 6: 1, 15: 1, 13: 1, 17: 2, 10: 2, 12: 2, 1: 2, 5: 2, 7: 2, 18: 2, 8: 3, 9: 3, 4: 4, 11: 4}
                    l5 = torch.zeros(5)
                    for i,_ in enumerate(label):
                        l5[inverted_nomenclature_five[i]] = l5[inverted_nomenclature_five[i]] or label[i]
                    label = l5
            
            yield (image, label)

    if "webdataset_root_path" not in configs or configs["webdataset_root_path"] is None:
        configs["webdataset_path"] = os.path.join(configs["root_path"], "webdataset", configs["dataset"])
    else:
        configs["webdataset_path"] = os.path.join(
            os.path.expandvars(configs["webdataset_root_path"]),
            "webdataset",
            configs["dataset"],
        )
    if not os.path.exists(configs["webdataset_path"]):
        raise FileNotFoundError(f"Webdataset path {configs['webdataset_path']} does not exist.")
    max_train_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "train", "*.tar")))[-1]
    max_train_index = max_train_shard.split("-train-")[-1][:-4]

    train_shards = os.path.join(
        configs["webdataset_path"],
        "train",
        "sample-train-{000000.." + max_train_index + "}.tar",
    )
      
    train_dataset = wds.WebDataset(train_shards, shardshuffle=True, resampled=False).shuffle(
        configs["webdataset_shuffle_size"]
    )
    train_dataset = train_dataset.compose(get_patches)
    train_dataset = train_dataset.batched(configs["batch_size"], partial=False)
   

    train_loader = wds.WebLoader(
        train_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],
        prefetch_factor=configs["prefetch_factor"],
        persistent_workers=configs["persistent_workers"],
    )
    train_loader = (
        train_loader.unbatched()
        .shuffle(
            configs["webdataset_shuffle_size"],
            initial=configs["webdataset_initial_buffer"],
        )
        .batched(configs["batch_size"])
    )
    if repeat:
        train_loader = train_loader.repeat()

    max_val_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "val", "*.tar")))[-1]
    max_val_index = max_val_shard.split("-val-")[-1][:-4]
    val_shards = os.path.join(
        configs["webdataset_path"],
        "val",
        "sample-val-{000000.." + max_val_index + "}.tar",
    )

    val_dataset = wds.WebDataset(val_shards, shardshuffle=False, resampled=False)
    val_dataset = val_dataset.compose(get_patches_eval)
    val_dataset = val_dataset.batched(configs["batch_size"], partial=True)

    val_loader = wds.WebLoader(
        val_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"]
    )
    max_test_shard = np.sort(glob.glob(os.path.join(configs["webdataset_path"], "test", "*.tar")))[-1]
    max_test_index = max_test_shard.split("-test-")[-1][:-4]
    test_shards = os.path.join(
        configs["webdataset_path"],
        "test",
        "sample-test-{000000.." + max_test_index + "}.tar",
    )

    test_dataset = wds.WebDataset(test_shards, shardshuffle=False, resampled=False)
    test_dataset = test_dataset.compose(get_patches_eval)
    test_dataset = test_dataset.batched(configs["batch_size"], partial=True)

    test_loader = wds.WebLoader(
        test_dataset,
        num_workers=configs["num_workers"],
        batch_size=None,
        shuffle=False,
        pin_memory=configs["pin_memory"],
        drop_last=False,
    )

    return train_loader, val_loader, test_loader