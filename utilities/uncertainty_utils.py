
from uncertainty import uncertainizer
import timm
from timm.models._pretrained import PretrainedCfg
from typing import Optional, Union, Dict, Any
from timm.models._helpers import load_checkpoint
import numpy as np

def add_unc_module(model, unc_module, unc_width, unc_depth=3, init_prednet_zero=False, stopgrad=False):
    if unc_module == "pred-net" or unc_module == "prednet":
        return uncertainizer.UncertaintyViaNetwork(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module == "pred-transformer":
        return uncertainizer.UncertaintyViaTransformer(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad,pool='mean')
    elif unc_module == "deep-prednet":
        return uncertainizer.UncertaintyViaDeepNet(model, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    elif unc_module.startswith("pred-net-layer_") or unc_module.startswith("prednet-layer_"):
        idxes = [int(unc_module.split("_")[-1])]
        return uncertainizer.UncertaintyViaDeepNet(model, hook_layer_idxes=idxes, width=unc_width, depth=unc_depth, init_prednet_zero=init_prednet_zero, stopgrad=stopgrad)
    else:
        raise NotImplementedError(f"Argument --unc_module {unc_module} is not implemented.")
    
def create_model(
        model_name: str,
        unc_module:str = "none",
        unc_width:int = 512,
        unc_depth:int = 3,
        pretrained: bool = False,
        pretrained_cfg: Optional[Union[str, Dict[str, Any], PretrainedCfg]] = None,
        pretrained_cfg_overlay:  Optional[Dict[str, Any]] = None,
        checkpoint_path: str = '',
        scriptable: Optional[bool] = None,
        exportable: Optional[bool] = None,
        no_jit: Optional[bool] = None,
        num_heads: int = 1,
        init_prednet_zero=False,
        stopgrad=False,
        uncertainty_checkpoint=None,
        **kwargs,
):
    model = timm.create_model(
        model_name,
        pretrained,
        pretrained_cfg,
        pretrained_cfg_overlay,
        checkpoint_path,
        scriptable,
        exportable,
        no_jit,
        **kwargs)

    model = add_unc_module(model, unc_module, unc_width, unc_depth, init_prednet_zero, stopgrad)
    
    if uncertainty_checkpoint is not None:
        load_checkpoint(model, uncertainty_checkpoint)

    return model

def build_unc_model_from_checkpoint(configs):
    model = timm.create_model(configs['model_name'], in_chans=configs["in_channels"], img_size=configs["image_size"],pretrained=False, num_classes=configs['pretrained_classes'],
                              checkpoint_path=configs['trained_model_path'], dynamic_img_pad=True, dynamic_img_size=True)
    model = add_unc_module(model, configs['unc_module'], configs['unc_width'], configs['unc_depth'], configs['init_prednet_zero'],configs['stopgrad'])
    return model