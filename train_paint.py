#=======[import default libraries]=======
import os
import pickle
import sys
from datetime import datetime

#=======[import libraries]=======
from diffusers import DDIMScheduler
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Resize, Normalize, ToTensor, Lambda
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor
import matplotlib.pyplot as plt

#=======[import own libraries]=======
from utils.dataset_paint import MyDataset
from utils.config_paint import TrainingConfig
from utils.check_gpu import display_gpu
from utils.get_model import getModel
from diffusers_lib.pipelines.controlnet.pipeline_controlnet import StableDiffusionControlNetPipeline
from diffusers_lib.models.controlnet import ControlNetModel


# set max_split_size_mb
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

#========[train]========
def train_loop(
        config, 
        unet, 
        controlnet, 
        vae, 
        text_encoder, 
        tokenizer,
        feature_extractor, 
        noise_scheduler, 
        optimizer,
        train_dataloader, 
        lr_scheduler, 
        device,
        dtype
    ):

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    # Now you train the model
    accumulation_steps = config.gradient_accumulation_steps

    for epoch in range(config.num_epochs):
        #for step, (cropped_frame1, cropped_frame2) in enumerate(train_dataloader):
        for step, batch in enumerate(train_dataloader):
            # 勾配をクリア
            if step % accumulation_steps == 0:
                optimizer.zero_grad()

            # Convert images to latent space
            latents = vae.encode(batch["pixel_value"].to(dtype=dtype, device=device)).latent_dist.sample()
            latents = latents * vae.config.scaling_factor

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get the text embedding for conditioning
            inputs = tokenizer(batch["prompt"], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").to(device=device)
            encoder_hidden_states = text_encoder(inputs.input_ids)[0]

            controlnet_image = batch["conditioning_pixel_value"].to(dtype=dtype, device=device)

            down_block_res_samples, mid_block_res_sample = controlnet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                controlnet_cond=controlnet_image,
                return_dict=False,
            )

            # Predict the noise residual
            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                down_block_additional_residuals=[
                    sample.to(dtype=dtype) for sample in down_block_res_samples
                ],
                mid_block_additional_residual=mid_block_res_sample.to(dtype=dtype),
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
            

            # スケールされた損失でバックワードパスを実行
            (loss / accumulation_steps).backward()

            # 一定のステップごとにパラメータを更新
            if (step + 1) % accumulation_steps == 0 or (step + 1 == len(train_dataloader)):
                torch.nn.utils.clip_grad_norm_(controlnet.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()  # 次の蓄積のために勾配をクリア


            # stepが100の倍数のときに進捗とlossを表示
            if step % 10 == 0:
                print(f"epoch: {epoch}, step: {step}/{len(train_dataloader)}, loss: {loss.item()}")        

        # epoch数が規定のものになったらモデルを保存する
        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # save model
            print("save model")
            save_dir = f"./output/{epoch}_paint"
            # ディレクトリが存在しない場合は作成
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # ファイルパスを設定
            save_path = os.path.join(save_dir, f"model.ckpt")

            # モデルを保存
            torch.save(controlnet, save_path)

def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # The inverse of normalize
    return tensor

def display_max_min(frame):
    # Calculate and print the max and min values of the frame
    max_val = frame.max()
    min_val = frame.min()
    print(min_val, max_val)

#========[main]========
def main():
    print("start!!!!")
    torch.cuda.empty_cache()
    #set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32 if device == torch.device('cuda') else torch.float32
    
    # import config
    config = TrainingConfig()


    # get dataset
    dataset = MyDataset()


    # set dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # import models
    unet = getModel("unet").to(device).to(dtype=dtype)
    controlnet = ControlNetModel.from_unet(unet).to(device).to(dtype=dtype) # 訓練対象
    #vae = getModel("vae").to(device).to(dtype=dtype)
    vae = AutoencoderKL.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/vae").to(device).to(dtype=dtype)
    noise_scheduler = DDIMScheduler.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/scheduler")
    tokenizer = CLIPTokenizer.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/tokenizer")
    text_encoder =  CLIPTextModel.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/text_encoder").to(device).to(dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained("/public/tsukaue/weights/stable-diffusion-2-1/feature_extractor")

    # to eval mode　
    controlnet.train()
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)



    # set optimizer
    optimizer = torch.optim.AdamW(
        controlnet.parameters(),
        lr=config.learning_rate,
        betas=(config.adam_beta1[0], config.adam_beta2[0]),
        weight_decay=config.adam_weight_decay[0],
        eps=config.adam_epsilon,   
    )
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(config, unet, controlnet, vae, text_encoder, tokenizer, feature_extractor, noise_scheduler, optimizer, train_dataloader, lr_scheduler, device, dtype)


if __name__ == "__main__":
    main()