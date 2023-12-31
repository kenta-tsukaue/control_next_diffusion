#=======[import default libraries]=======
import io
import os
import pickle
import sys
from datetime import datetime

#=======[import libraries]=======
from diffusers import DDIMScheduler, AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import matplotlib.pyplot as plt

#=======[import own libraries]=======
from utils.loss import get_loss
from utils.dataset import CustomDataset
from utils.dataset_paint import MyDataset
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
        #Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    #========[predict]========
    pipe.enable_attention_slicing()
    image_path="golf.jpg"

    # PILを使って画像を読み込む
    image = Image.open(image_path)

    # 画像を変換
    tensor_image = transform(image)
    print(tensor_image)

    # バッチ次元を追加
    tensor_image = tensor_image.unsqueeze(0)
    
    prompt = [""] 
    output = pipe( prompt=prompt, image=tensor_image, guess_mode=True, do_classifier_free_guidance=True)

    image_data = output[0][0]
    image_data.save(output_path)

    """正規化を直す
    image_array = np.array(image_data)  # PIL画像をnumpy配列に変換
    
    # 定義済みの平均と標準偏差
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    # 画像データの正規化を解除
    image_array = (image_array / 255 - mean) / std  # 正規化解除の計算

    image_array = np.clip(image_array * 255, 0, 255).astype(np.uint8)

    # numpy配列をPIL画像に戻す
    unnormalized_image = Image.fromarray(image_array)

    unnormalized_image.save(output_path)"""


def main():

    # import models
    unet = getModel("unet").to(device).to(dtype=dtype)
    controlnet = torch.load("output/59/model.ckpt").to(device).to(dtype=dtype)
    #controlnet = ControlNetModel.from_unet(unet).to(device).to(dtype=dtype)
    vae = AutoencoderKL.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/vae").to(device).to(dtype=dtype)
    noise_scheduler = DDIMScheduler.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/scheduler", subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/tokenizer")
    text_encoder =  CLIPTextModel.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/text_encoder").to(device).to(dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/feature_extractor")

    # to eval mode　
    controlnet.requires_grad_(False).eval()
    unet.requires_grad_(False).eval()
    vae.requires_grad_(False).eval()
    text_encoder.requires_grad_(False).eval()

    predict(vae, text_encoder, tokenizer, unet, controlnet, noise_scheduler, feature_extractor,"output/59/sample.png")

if __name__ == "__main__":
    main()