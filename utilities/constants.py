PRETRAINED_IMAGENET_CONFIGS = {
    "vit-tiny":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"uncertainty_checkpoints/vit_tiny_checkpoint.pth.tar"
    },
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"uncertainty_checkpoints/vit_small_checkpoint.pth.tar"
    },
    "vit-base":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"uncertainty_checkpoints/vit_base_checkpoint.pth.tar"
    },
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"uncertainty_checkpoints/vit_large_checkpoint.pth.tar",
    },
}

PRETRAINED_BIGEARTHNET_CONFIGS = {
    "vit-tiny":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/bigearthnet/best_model.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/bigearthnet/best_model.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/bigearthnet/best_model.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_BIGEARTHNET5_CONFIGS = {
    "vit-tiny":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/bigearthnet/best_model_5.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":5,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/bigearthnet/best_model_5.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":5,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/bigearthnet/best_model_5.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":5,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_5.pth",
        "in_chans":3,
        "img_size":120,
        "num_classes":5,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}


PRETRAINED_BIGEARTHNET_SAR_CONFIGS = {
    "vit-tiny":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/bigearthnet/best_model_sar.pth",
        "in_chans":2,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/bigearthnet/best_model_sar.pth",
        "in_chans":2,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/bigearthnet/best_model_sar.pth",
        "in_chans":2,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_sar.pth",
        "in_chans":2,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_BIGEARTHNET_MS_CONFIGS = {
    "vit-tiny":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/bigearthnet/best_model_ms.pth",
        "in_chans":12,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/bigearthnet/best_model_ms.pth",
        "in_chans":12,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/bigearthnet/best_model_ms.pth",
        "in_chans":12,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_ms.pth",
        "in_chans":12,
        "img_size":120,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_BIGEARTHNET19_16_CONFIGS = {
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_19_16.pth",
        "in_chans":3,
        "img_size":16,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_BIGEARTHNET19_30_CONFIGS = {
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_19_30.pth",
        "in_chans":3,
        "img_size":30,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_BIGEARTHNET19_60_CONFIGS = {
    "vit-large":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/bigearthnet/best_model_19_60.pth",
        "in_chans":3,
        "img_size":60,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_SEN12MS_CONFIGS = {
    "vit-tiny":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/sen12ms/best_model.pth",
        "in_chans":3,
        "img_size":256,
        "num_classes":17,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 512,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/sen12ms/best_model.pth",
        "in_chans":3,
        "img_size":256,
        "num_classes":17,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/sen12ms/best_model.pth",
        "in_chans":3,
        "img_size":256,
        "num_classes":17,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/sen12ms/best_model.pth",
        "in_chans":3,
        "img_size":256,
        "num_classes":17,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}

PRETRAINED_FLAIR_CONFIGS = {
    "vit-tiny":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_tiny_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_tiny/flair/best_model.pth",
        "in_chans":3,
        "img_size":224,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-small":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_small_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_small/flair/best_model.pth",
        "in_chans":3,
        "img_size":224,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }, 
    "vit-base":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_base_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_base/flair/best_model.pth",
        "in_chans":3,
        "img_size":224,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    },
    "vit-large":{
        "unc_width": 256,
        "unc_depth": 2,
        "model_name":"vit_large_patch16_224.augreg_in21k",
        "unc_module": "prednet",
        "stopgrad":True,
        "pretrained": True,
        "uncertainty_checkpoint":"trained_models/vit_large/flair/best_model.pth",
        "in_chans":3,
        "img_size":224,
        "num_classes":19,
        "dynamic_img_pad":True, 
        "dynamic_img_size":True
    }}