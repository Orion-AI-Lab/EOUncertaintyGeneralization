import torch.nn as nn
import torch
from torchvision import transforms
import tqdm 
import os
import numpy as np
from metrics.evaluation_metrics import recall_at_one, closest_representations
from torchmetrics.functional.classification import binary_auroc as auroc
from utilities import utils
from uncertainty.uncertainizer import EmbeddingTensor
from utilities.uncertainty_utils import build_unc_model_from_checkpoint, add_unc_module
from metrics.cpa import calc_cpa
from metrics.utils import AverageMeter, pct_cropped_has_bigger_uncertainty, save_to_csv, entropy_dispersion
import wandb
from losses.loss import LossPrediction, LossOrderLoss
import torchvision.transforms.functional as func_transforms
import timm
from utilities import constants
import segmentation_models_pytorch as smp

def inference(model,test_loader,device='cpu',save_features_path=None,loss_fn=None,recall_type='multiclass',load_features=False, dataset="",configs=None, recall_criterion='one',mode=None,train_dataset=None, task_predictor=None, task_criterion=None):
    if mode is None:
        raise ValueError("Mode must be provided")
    if save_features_path is None or not os.path.isfile(os.path.join(save_features_path,"all_features.npy")) or not load_features:
        model.eval()
        
        if configs is not None and configs['webdataset']:
            if dataset not in configs["num_samples_per_dataset"] or mode not in configs["num_samples_per_dataset"][dataset]:                
                '''TODO change implementation to incrementally build the features'''
                print(f"Calculating number of samples in the dataset {configs['dataset']}")
                n_data = 0
                for idx, batch in enumerate(tqdm.tqdm(test_loader)):
                    if configs["unc_module"] == "pred-transformer" and load_features:
                        img, tokens, label = batch
                    else:
                        img, label = batch
                    n_data += img.shape[0]
            else:
                n_data = configs["num_samples_per_dataset"][dataset][mode]
            #n_data = 269568
        else:
            n_data = len(test_loader.dataset)
            
        print(f"Number of samples in the dataset {dataset}: {n_data}")
        if configs["unc_module"] == "pred-transformer" and load_features:
            target_shape = next(iter(test_loader))[2].shape[1:]
        else:
            target_shape = next(iter(test_loader))[1].shape[1:]
        all_features = np.zeros((n_data, model.num_features), dtype="float32")
        all_targets = np.zeros((n_data, *target_shape), dtype="int")
        all_uncertainties = torch.zeros(n_data)
        all_moran = np.zeros((n_data))
        all_uncertainties_c = torch.zeros(n_data)
        if task_predictor is not None:
            all_losses = []
            #all_predictions = [] 
        cur_idx = 0
        
        cropped_has_bigger_unc = AverageMeter()
        if configs["compare_with_cropped"] == "cropping":
            center_crop = np.random.random(n_data) * (configs["max_cropping"] - configs["min_cropping"] ) + configs["min_cropping"] 

        for idx, batch in enumerate(tqdm.tqdm(test_loader)):
            if configs["unc_module"] == "pred-transformer" and load_features:
                image, tokens, label = batch
            else:
                image, label = batch
            if configs["compare_with_cropped"]:
                moran = np.zeros((image.shape[0]))
                for i in range(image.shape[0]):
                    if configs['recall_per_dataset'][dataset] == 'segmentation':
                        moran[i] = entropy_dispersion(label[i], patch_size=64)
                    if configs["compare_with_cropped"] == "gaussian_noise":
                        image_c[i] = image[i] + torch.randn_like(image[i]) * configs["gaussian_std"]
                    elif configs["compare_with_cropped"] == "cropping":
                        crop_size = int(round(min(image.shape[2], image.shape[3]) * center_crop[cur_idx + i])) 
                        image_c[i] = func_transforms.resize(func_transforms.center_crop(image[i], [crop_size]),
                                                            [image.shape[2], image.shape[3]])
                    else:
                        raise ValueError(f"Method {configs['compare_with_cropped']} is not implemented.")
                    image_c = image_c.to(device)
            
            if load_features:
                image = EmbeddingTensor(image)
                if configs["unc_module"] == "pred-transformer":
                    tokens = EmbeddingTensor(tokens)
                    tokens = tokens.to(device)
            image = image.to(device)
            label = label.to(torch.float).to(device)

            with torch.no_grad():
                if configs["unc_module"] == "pred-transformer" and load_features:
                    representation, uncertainty, features = model(image, tokens)
                else:
                    representation, uncertainty, features = model(image)
                
                all_uncertainties[cur_idx:(cur_idx + image.shape[0])] = uncertainty.detach().cpu().squeeze()
                all_features[cur_idx:(cur_idx + image.shape[0]),:] = features.detach().cpu().squeeze()
                all_targets[cur_idx:(cur_idx + image.shape[0])] = label.detach().cpu().squeeze()
                
                if task_predictor is not None:
                    out = task_predictor(image)
                    if configs["recall_per_dataset"][dataset] == "multilabel":
                        task_loss = task_criterion(out, label)
                    else:
                        task_loss = task_criterion(out,label.long())
                    all_losses.append(task_loss.detach().cpu().squeeze())
                    if configs["recall_per_dataset"][dataset] == "multilabel":
                        predictions = torch.sigmoid(out)
                        predictions[predictions >= 0.5] = 1
                        predictions[predictions < 0.5] = 0
                    else:
                        predictions = torch.argmax(out, dim=1)

                if configs["compare_with_cropped"]:
                    _, uncertainty_c, _ = model(image_c)
                    cBiggerUnc = pct_cropped_has_bigger_uncertainty(uncertainty.detach(), uncertainty_c.detach())
                    cropped_has_bigger_unc.update(cBiggerUnc.item(), image.size(0))
                    all_uncertainties_c[cur_idx:(cur_idx + image.shape[0])] = uncertainty_c.detach().cpu().squeeze()
                    all_moran[cur_idx:(cur_idx + image.shape[0])] = moran

            if dataset == train_dataset:
                try:
                    loss = loss_fn(representation, uncertainty, label, features, model.get_classifier())
                except Exception as e:
                    print(e)
                    print("Error in loss calculation. Removing last sample: ")
                    print("Initial shape: ", representation.shape)
                    representation = representation[:-1]
                    uncertainty = uncertainty[:-1]
                    features = features[:-1]
                    label = label[:-1]
                    print("Final shape: ", representation.shape)
                    loss = loss_fn(representation, uncertainty, label, features, model.get_classifier())
                if idx % 100 == 0:
                    log_dict = {f"val loss": loss.item()}
                    
                    if configs["wandb"]:
                        wandb.log(log_dict)

                    print(log_dict)

            cur_idx += image.shape[0]

        os.makedirs(os.path.join("density_plot_benet", f"{dataset}"), exist_ok=True)
        torch.save(all_uncertainties,os.path.join(f"density_plot_benet/{dataset}","all_uncertainties.pt"))

        if task_predictor is not None:
            all_losses = torch.cat(all_losses, dim=0)
            all_losses = all_losses.cpu().numpy()

            
            os.makedirs(os.path.join(save_features_path,"task_losses",configs['dataset'],'pretrained_on_'+configs['pretraining_dataset'],configs['model']), exist_ok=True)
            torch.save(all_losses,os.path.join(save_features_path,"task_losses",configs['dataset'],'pretrained_on_'+configs['pretraining_dataset'],configs['model'],f"all_losses_{mode}.pt"),pickle_protocol=4)
            torch.save(all_uncertainties,os.path.join(save_features_path,"task_losses",configs['dataset'],'pretrained_on_'+configs['pretraining_dataset'],configs['model'],f"all_uncertainties_{mode}.pt"),pickle_protocol=4)
            torch.save(all_targets,os.path.join(save_features_path,"task_losses",configs['dataset'],'pretrained_on_'+configs['pretraining_dataset'],configs['model'],f"all_targets_{mode}.pt"),pickle_protocol=4)
            return
        if save_features_path is not None and not os.path.isfile(os.path.join(save_features_path,"all_features.pt")):
            #Create the parent directory if it does not exist
            os.makedirs(save_features_path, exist_ok=True)
            np.save(os.path.join(save_features_path,"all_features.npy"), all_features)
            np.save(os.path.join(save_features_path,"all_targets.npy"), all_targets)
            torch.save(all_uncertainties,os.path.join(save_features_path,"all_uncertainties.pt"))
    else:
        all_features = np.load(os.path.join(save_features_path,"all_features.npy"))
        all_targets = np.load(os.path.join(save_features_path,"all_targets.npy"))
        all_uncertainties = torch.load(os.path.join(save_features_path,"all_uncertainties.pt"),weights_only=True)

    num_classes = all_targets.max() + 1

    print("Evaluating uncertainties on: ", dataset)
    print("Recall type: ", recall_type)

    if configs["compare_with_cropped"]:
        print("Mean uncertainties", all_uncertainties_c.mean())
        print('Mean Entropy ', all_moran.mean())
    
    cl_representations = closest_representations(all_features, all_targets, mode="faiss",device=device)

    recall = {}
    auroc_correct = {}

    for rc in recall_criterion[dataset]:
        recall[rc], correctness = recall_at_one(cl_representations, all_targets, type=recall_type, num_classes=num_classes, recall_criterion=rc)
        if (recall_type == 'segmentation' and rc in ["distribution_of_classes", "distribution_of_classes_with_patching"]) or (recall_type == 'multilabel' and rc == 'distance'):
            auroc_correct[rc] = calc_cpa( - all_uncertainties, torch.from_numpy(correctness)).item()
        else:
            auroc_correct[rc] = auroc( -all_uncertainties, torch.from_numpy(correctness).int()).item()
        

    return recall, auroc_correct

def infer(configs):
    #Get the pretrained model
    if configs['task'] == "inference":
        model = utils.get_pretrained_model(configs)
    else:
        model = build_unc_model_from_checkpoint(configs)
    
    #Create the data loader
    _,_, test_loader = utils.get_loaders(configs['dataset'], download=True, dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)
    #Infere the uncertainties
    recall, auroc = inference(model, test_loader, configs['device'], save_features_path=configs['save_features_path'],dataset=configs['dataset'],
                            recall_type=configs['recall_per_dataset'][configs['dataset']], recall_criterion=configs['recall_criterion_per_dataset'],configs=configs,mode="test")    
    for rc in auroc:
        print("Recall criterion: ", rc)
        print("Recall at one: ", recall[rc])
        print("R-AUROC: ", auroc[rc])
    
def save_features(configs):
    #Get the pretrained model
    model = build_unc_model_from_checkpoint(configs)

    model.set_grads(backbone=False, classifier=not configs['freeze_classifier'], unc_module=True)
    model.to(configs['device'])
    #Create the data loader
    configs['shuffle'] = False
    train_loader, val_loader, test_loader = utils.get_loaders(configs['dataset'], download=True, features_root=None,dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)
    model_name = "_".join(configs['model_name'].split("_")[:2])
    if configs["loss"] == "loss_prediction":
        loss_fn = LossPrediction(lambda_=configs["lambda_value"], inv_temp=configs["inv_temp"], ignore_ce_loss=configs["freeze_classifier"])
    elif configs["loss"] == "order_loss":
        loss_fn = LossPrediction(lambda_=configs["lambda_value"], inv_temp=configs["inv_temp"], unc_loss=LossOrderLoss(), ignore_ce_loss=configs["freeze_classifier"])
    else:
        raise NotImplementedError(f"Loss {configs['loss']} is not implemented.")
    
    #Infere the uncertainties
    for loader, mode in zip([train_loader, val_loader, test_loader], ["train", "val", "test"]):
        print("Saving features for: ", mode)
        save_features_path = os.path.join(configs['save_features_path'],model_name,configs['dataset'],mode)
        recall, auroc = inference(model, loader, configs['device'], save_features_path=save_features_path,loss_fn=loss_fn, dataset=configs['dataset'],recall_type=configs['recall_per_dataset'][configs['dataset']],
                                  recall_criterion=configs['recall_criterion_per_dataset'],configs=configs,mode=mode, train_dataset=configs['dataset'])
        
        print("Evaluating uncertainties on: ", configs['dataset'])
        print("Recall at one: ", recall)
        print("R-AUROC: ", auroc)
        
        
def uncertainty_vs_loss(configs):
    model = utils.get_pretrained_model(configs)
    if configs['dataset'] == 'sen12ms':
        task_configs = constants.PRETRAINED_SEN12MS_CONFIGS[configs['model']]
        task_configs['init_prednet_zero'] = False
    elif configs['dataset'] == 'bigearthnet':
        task_configs = constants.PRETRAINED_BIGEARTHNET_CONFIGS[configs['model']]
        task_configs['init_prednet_zero'] = False
    else:
        task_configs = None
    if configs['recall_per_dataset'][configs['dataset']] == "segmentation":
        task_predictor = smp.Unet(
            encoder_name="resnet50",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=configs["task_in_channels"],  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=configs["task_num_classes"],  # model output channels (number of classes in your dataset)
        )
    else:
        task_predictor = timm.create_model(constants.PRETRAINED_IMAGENET_CONFIGS[configs['model']]["model_name"], in_chans=configs['task_in_channels'], num_classes=configs["task_num_classes"], img_size=configs["task_image_size"], pretrained=False)
    if task_configs is not None:
        task_predictor = add_unc_module(task_predictor, task_configs['unc_module'], task_configs['unc_width'], task_configs['unc_depth'], task_configs['init_prednet_zero'],task_configs['stopgrad'])
        task_predictor.load_state_dict(torch.load(configs['task_predictor_weights_path']))
        task_predictor = task_predictor.model
    else:
        task_predictor.load_state_dict(torch.load(configs['task_predictor_weights_path']))
    task_predictor.to(configs['device'])
    
    #Create the data loader
    _,_, test_loader = utils.get_loaders(configs['dataset'], download=True, dataset_root=configs['dataset_root_path'],batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True,configs=configs)
    
    if configs['recall_per_dataset'][configs['dataset']] == "multilabel":
        task_criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        task_criterion = nn.CrossEntropyLoss(reduction='none')
    
    #Infere the uncertainties
    recall, auroc = inference(model, test_loader, configs['device'], save_features_path=configs['save_features_path'],dataset=configs['dataset'],
                            recall_type=configs['recall_per_dataset'][configs['dataset']], recall_criterion=configs['recall_criterion_per_dataset'],configs=configs,mode="test", task_predictor=task_predictor, task_criterion=task_criterion)    
    for rc in auroc:
        print("Recall criterion: ", rc)
        print("Recall at one: ", recall[rc])
        print("R-AUROC: ", auroc[rc])