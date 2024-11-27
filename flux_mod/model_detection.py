import logging
import torch
from comfy.model_detection import detect_unet_config
from .supported_models import Flux, FluxSchnell
import comfy.supported_models_base

models = { Flux, FluxSchnell }

def model_config_from_unet_config(unet_config, state_dict=None):
    for model_config in models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)

    logging.error("no match {}".format(unet_config))
    return None

def model_config_from_unet(state_dict, unet_key_prefix, use_base_if_no_match=False):
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    if unet_config is None:
        return None
    model_config = model_config_from_unet_config(unet_config, state_dict)
    if model_config is None and use_base_if_no_match:
        model_config = comfy.supported_models_base.BASE(unet_config)

    scaled_fp8_key = "{}scaled_fp8".format(unet_key_prefix)
    if scaled_fp8_key in state_dict:
        scaled_fp8_weight = state_dict.pop(scaled_fp8_key)
        model_config.scaled_fp8 = scaled_fp8_weight.dtype
        if model_config.scaled_fp8 == torch.float32:
            model_config.scaled_fp8 = torch.float8_e4m3fn

    return model_config
