#=======[import default libraries]=======
import io
import os
import pickle
import sys
from datetime import datetime

#=======[import libraries]=======
from diffusers import DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

#=======[import own libraries]=======
from utils.loss import get_loss
from utils.dataset import CustomDataset
from utils.config import TrainingConfig
from utils.check_gpu import display_gpu
from utils.get_model import getModel
from diffusers_lib.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers_lib.models.controlnet import ControlNetModel

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float32 if device == torch.device('cuda') else torch.float32


def predict(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, feature_extractor, output_path):
    # set pipline
    pipe = StableDiffusionControlNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        controlnet=controlnet,
        scheduler=noise_scheduler,
        safety_checker=None,
        feature_extractor=feature_extractor,
    )

    # set transform
    transform = Compose([
        ToTensor(),
        Resize((768, 768)),  # 768x768にリサイズ
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #========[predict]========
    pipe.enable_attention_slicing()
    image_path="golf.jpg"

    # PILを使って画像を読み込む
    image = Image.open(image_path)

    # 画像を変換
    tensor_image = transform(image)

    # バッチ次元を追加

    tensor_image = tensor_image.unsqueeze(0)
    prompt = [""] 
    output = pipe( prompt=prompt, image=tensor_image)

    image_data = output[0][0]

    image_data.save(output_path)
    
    del pipe

def main():

    # import models
    unet = getModel("unet").to(device).to(dtype=dtype)
    controlnet = torch.load("weights/20231221_025011.ckpt").to(device).to(dtype=dtype)
    #controlnet = ControlNetModel.from_unet(unet).to(device).to(dtype=dtype)
    vae = getModel("vae").to(device).to(dtype=dtype)
    noise_scheduler = DDIMScheduler.from_pretrained("weights/stable-diffusion-2-1/scheduler", subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained("weights/stable-diffusion-2-1/tokenizer")
    text_encoder =  CLIPTextModel.from_pretrained("weights/stable-diffusion-2-1/text_encoder").to(device).to(dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained("weights/stable-diffusion-2-1/feature_extractor")

    # to eval mode　
    controlnet.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    predict(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, feature_extractor)

if __name__ == "__main__":
    main()