#=======[import default libraries]=======
import os
import pickle
import sys

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
        criterion,
        train_dataloader, 
        lr_scheduler, 
        device,
        dtype
    ):

    if config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader))
        progress_bar.set_description(f"Epoch {epoch}")

        for step, (cropped_frame1, cropped_frame2) in enumerate(train_dataloader):
            prompt = [""] * config.train_batch_size
            cropped_frame1 = cropped_frame1.to(device)
            cropped_frame2 = cropped_frame2.to(device)

            nan_in_controlnet_weights = any(torch.isnan(param).any() for param in controlnet.parameters())
            print(" nan_in_controlnet_weights",  nan_in_controlnet_weights)
            
            optimizer.zero_grad()
            # get loss
            pred, noise = get_loss(
                unet,
                vae,
                controlnet,
                noise_scheduler,
                text_encoder,
                tokenizer,
                feature_extractor,
                device,
                dtype,
                prompt,
                cropped_frame1,
                cropped_frame2,
                do_classifier_free_guidance=False
            )
            #loss = criterion(pred, noise)
            loss = F.mse_loss(pred, noise)
            print(loss)
            print(loss.requires_grad)
            loss.backward()
            optimizer.step()
            #lr_scheduler.step()
            progress_bar.update(1)


        
        if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
            pipeline = StableDiffusionControlNetPipeline(
                vae,
                text_encoder,
                tokenizer,
                unet,
                controlnet,
                noise_scheduler,
                feature_extractor
            )
            evaluate(config, epoch, pipeline)

        if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
            # save model
            print("save model")




#========[evaluate]========
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.manual_seed(config.seed),
    ).images


#========[main]========
def main():
    torch.cuda.empty_cache()
    #set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float16 if device == torch.device('cuda') else torch.float32

    # import config
    config = TrainingConfig()


    # set transform
    transform = Compose([
        Resize((768, 768)),  # 768x768にリサイズ
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # get dataset
    dataset = CustomDataset(config, device, transform=transform)


    # set dataloader
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True)

    # import models
    unet = getModel("unet").to(device).to(dtype=dtype)
    controlnet = ControlNetModel.from_unet(unet).to(device).to(dtype=dtype) # 訓練対象
    vae = getModel("vae").to(device).to(dtype=dtype)
    noise_scheduler = DDIMScheduler.from_pretrained("weights/stable-diffusion-2-1/scheduler", subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained("weights/stable-diffusion-2-1/tokenizer")
    text_encoder =  CLIPTextModel.from_pretrained("weights/stable-diffusion-2-1/text_encoder").to(device).to(dtype=dtype)
    feature_extractor = CLIPImageProcessor.from_pretrained("weights/stable-diffusion-2-1/feature_extractor")

    # to eval mode　
    controlnet.train()
    unet.eval()
    vae.eval()
    text_encoder.eval()

    # set optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(controlnet.parameters(), lr=config.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    train_loop(config, unet, controlnet, vae, text_encoder, tokenizer, feature_extractor, noise_scheduler, optimizer, criterion, train_dataloader, lr_scheduler, device, dtype)


if __name__ == "__main__":
    main()