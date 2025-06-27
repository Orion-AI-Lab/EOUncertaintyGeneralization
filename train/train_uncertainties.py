import torch
import torch.nn as nn
import tqdm
import wandb

from utilities.uncertainty_utils import build_unc_model_from_checkpoint
from utilities.utils import get_loaders, adjust_learning_rate
from losses.loss import LossPrediction, LossOrderLoss
from inference.infere_uncertainties import inference
import os
from uncertainty.uncertainizer import EmbeddingTensor
import datetime

def train_one_epoch(model, train_loader, optimizer, loss_fn, device, epoch,scaler,load_features,configs):
    model.train()
    total_loss = 0
    total_samples = 0.0
    for idx, batch in enumerate(tqdm.tqdm(train_loader)):
        optimizer.zero_grad()
        with torch.amp.autocast(configs['device'],enabled=configs["mixed_precision"]):
            if configs["unc_module"] == "pred-transformer" and load_features:
                image, tokens, label = batch
            else:
                image, label = batch
            if load_features:
                image = EmbeddingTensor(image)
                if configs["unc_module"] == "pred-transformer":
                    tokens = EmbeddingTensor(tokens)
                    tokens = tokens.to(configs["device"])
            image = image.to(configs["device"])
            label = label.to(configs["device"]).float()
            total_samples += image.shape[0]

            if configs["unc_module"] == "pred-transformer" and load_features:
                output, uncertainties, features = model(image, tokens)
            else:
                output, uncertainties, features= model(image)

            loss = loss_fn(output, uncertainties, label, features, model.get_classifier())
            
            if idx % 100 == 0:
                log_dict = {"Epoch": epoch, "Iteration": idx, "train loss": loss.item(),"Learning rate":optimizer.param_groups[0]['lr']}
                
                if configs["wandb"]:
                    wandb.log(log_dict)

                print(log_dict)
                    
        if configs["mixed_precision"]:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if configs['lr_schedule'] == "cosine":
            if "num_samples_per_dataset" not in configs:
                num_steps = len(train_loader)
            else:
                num_steps = configs["num_samples_per_dataset"][configs["dataset"]]["train"] // configs["batch_size"]
            adjust_learning_rate(optimizer, idx / num_steps + epoch, configs)

    return total_loss/total_samples

def train(configs):
    model = build_unc_model_from_checkpoint(configs)

    model.set_grads(backbone=False, classifier=not configs['freeze_classifier'], unc_module=True)

    val_loader = {}
    test_loader = {}
    test_configs = {}

    train_loader, val_loader[configs['dataset']], test_loader[configs['dataset']] = get_loaders(configs['dataset'], download=False, dataset_root=configs['dataset_root_path'],features_root=configs['features_root_path'],
                                                        batch_size=configs['batch_size'], num_workers=configs['num_workers'], pin_memory=True, feature_load=configs['feature_loader'], configs=configs)
    
    test_configs[configs['dataset']] = configs.copy()
    test_configs[configs['dataset']]["recall_type"] = configs["recall_per_dataset"][configs['dataset']]

    for dataset in configs['test_datasets']:
        tmp_configs = configs.copy()
        tmp_configs['dataset'] = dataset
        tmp_configs["feature_loader"] = False
        tmp_configs["webdataset"]=tmp_configs["test_webdatasets"]
        tmp_configs["eval_webdataset"]=tmp_configs["webdataset"]
        tmp_configs["recall_type"] = tmp_configs["recall_per_dataset"][dataset]
        test_configs[dataset] = tmp_configs
        val_loader[dataset] = get_loaders(tmp_configs['dataset'], download=False, dataset_root=tmp_configs['dataset_root_path'],features_root=tmp_configs['features_root_path'],
                                                    batch_size=tmp_configs['batch_size'], num_workers=tmp_configs['num_workers'], pin_memory=True, 
                                                    feature_load=tmp_configs['feature_loader'], configs=tmp_configs)[1]
        test_loader[dataset] = get_loaders(tmp_configs['dataset'], download=False, dataset_root=tmp_configs['dataset_root_path'],features_root=tmp_configs['features_root_path'],
                                                    batch_size=tmp_configs['batch_size'], num_workers=tmp_configs['num_workers'], pin_memory=True, 
                                                    feature_load=tmp_configs['feature_loader'], configs=tmp_configs)[2]

    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.amp.GradScaler(device=configs["device"])
    else:
        scaler = None
        
    #Create loss
    if configs["loss"] == "loss_prediction":
        loss_fn = LossPrediction(lambda_=configs["lambda_value"], inv_temp=configs["inv_temp"], ignore_ce_loss=configs["freeze_classifier"])
    elif configs["loss"] == "order_loss":
        loss_fn = LossPrediction(lambda_=configs["lambda_value"], inv_temp=configs["inv_temp"], unc_loss=LossOrderLoss(), ignore_ce_loss=configs["freeze_classifier"])
    else:
        raise NotImplementedError(f"Loss {configs['loss']} is not implemented.")
    
    optimizer =  torch.optim.AdamW(model.parameters(), lr=configs['lr'], weight_decay=configs['weight_decay'])
    
    #Create ckpt dir if it does not exist
    now = datetime.datetime.now()
    name = "_".join(configs['model_name'].split("_")[:2])
    ckpt_path = f"train_unc_checkpoints/{name}/{configs['dataset']}/{now}"
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path, exist_ok=True)
    model.to(configs['device'])
    best_auroc = 0.0
    for epoch in range(configs['epochs']):
        recall = {}
        auroc = {}
        train_one_epoch(model, train_loader, optimizer, loss_fn, configs['device'],epoch,scaler,configs['feature_loader'],configs)
        for dataset in [configs['dataset']] + configs['test_datasets']:
            recall[dataset], auroc[dataset] = inference(model, val_loader[dataset], configs['device'], save_features_path=None, loss_fn = loss_fn, dataset=dataset,
                                                        load_features=test_configs[dataset]['feature_loader'], recall_type=test_configs[dataset]['recall_type'], 
                                                        recall_criterion=configs['recall_criterion_per_dataset'],configs=test_configs[dataset], mode="val",
                                                        train_dataset=configs['dataset'])
            for rc in auroc[dataset]:
                log_dict = {f"Val Recall@1 {dataset} for RC {rc}": recall[dataset][rc], f"Val R-Auroc {dataset} for RC {rc}": auroc[dataset][rc]}
                if configs["wandb"]:
                    wandb.log(log_dict)
                print(log_dict)

        if auroc[configs['dataset']][configs['val_recall_criterion']]>best_auroc:
            print(f"Validation for {configs['dataset']} with recall criterion {configs['val_recall_criterion']}")
            best_auroc = auroc[configs['dataset']][configs['val_recall_criterion']]
            print("New best model found with auroc/recall: ", best_auroc, recall[configs['dataset']])
            torch.save(model.state_dict(),ckpt_path + f"/best_model_{epoch}.pth")
    
    recall = {}
    auroc = {}

    for dataset in [configs['dataset']] + configs['test_datasets']:
        recall[dataset], auroc[dataset] = inference(model, test_loader[dataset], configs['device'], save_features_path=None, dataset=dataset,load_features=test_configs[dataset]['feature_loader'],
                            recall_type=test_configs[dataset]['recall_type'], recall_criterion=configs['recall_criterion_per_dataset'],configs=test_configs[dataset], mode="test") 

        print("Evaluating uncertainties on: ", dataset)
        log_dict = {f"Test Recall@1 {dataset} for RC {rc}": recall[dataset][rc], f"Test R-Auroc {dataset} for RC {rc}": auroc[dataset][rc]}
        if configs["wandb"]:
            wandb.log(log_dict)
        print(log_dict)
        
    