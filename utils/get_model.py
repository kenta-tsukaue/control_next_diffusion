import json
import sys
import os
from typing import Union
from diffusers import AutoencoderKL
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection
from safetensors import safe_open


from diffusers_lib.models.unet_2d_condition import UNet2DConditionModel



# model information
model_dict = {
    "unet": {"model":UNet2DConditionModel, "config_path": "config.json", "exist_safetensor_file":True},
    "vae": {"model":AutoencoderKL, "config_path": "config.json", "exist_safetensor_file":True},
    "feature_extractor": {"model":CLIPImageProcessor, "config_path": "preprocessor_config.json", "exist_safetensor_file":False},
    "image_encoder": {"model":CLIPVisionModelWithProjection, "config_path": "config.json", "exist_safetensor_file":True},
}

def getModel(key:str) -> Union[
        UNet2DConditionModel, 
        AutoencoderKL, 
        CLIPImageProcessor,
        CLIPVisionModelWithProjection
    ]:

    # read config
    config_path = "weights/stable-diffusion-2-1/" + key + "/" + model_dict[key]["config_path"]
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # set model
    model = model_dict[key]["model"](**config)

    # check if the model is pre trained
    if model_dict[key]["exist_safetensor_file"]:
        safetensor_file_path = "weights/stable-diffusion-2-1/" + key + "/diffusion_pytorch_model.fp16.safetensors"
        tensors = {}

        # read safetensors
        with safe_open(safetensor_file_path, framework="pt") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        
        # set parameters
        model.load_state_dict(tensors)
    
    return model