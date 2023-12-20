import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F

from diffusers_lib.image_processor import VaeImageProcessor


def get_loss(
    unet,
    vae,
    controlnet,
    noise_scheduler,
    text_encoder,
    tokenizer,
    feature_extractor,
    device,
    dtype,
    prompt: Union[str, List[str]] = None,
    image = None,
    image_c = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    guidance_scale: float = 7.5,
    num_images_per_prompt: Optional[int] = 1,
    latents: Optional[torch.FloatTensor] = None,
    guess_mode: bool = False,
    do_classifier_free_guidance = True,
):
    with torch.no_grad():
        # 0. Settings
        batch_size = len(prompt)
        vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
        height = height or unet.config.sample_size * vae_scale_factor
        width = width or unet.config.sample_size * vae_scale_factor
        control_image_processor = VaeImageProcessor(
            vae_scale_factor=vae_scale_factor, do_normalize=False
        )
        image = image.to(device=device)
        image_c = image_c.to(device=device)

        
        # 1. Encode input prompt
        prompt_embeds, negative_prompt_embeds = encode_prompt(
            unet,
            text_encoder,
            tokenizer,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 2. Prepare image_c
        image_c = prepare_image(
            image=image_c,
            width=width,
            height=height,
            batch_size=batch_size * num_images_per_prompt,
            control_image_processor=control_image_processor,
            num_images_per_prompt=num_images_per_prompt,
            device=device,
            dtype=controlnet.dtype,
            do_classifier_free_guidance=do_classifier_free_guidance,
            guess_mode=guess_mode,
        )


        image = image.to(dtype=dtype)
        image_c = image_c.to(dtype=dtype)


        # 4. Prepare noises
        num_channels_latents = unet.config.in_channels
        noise = prepare_noise(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            vae_scale_factor,
        )

        # 5. Prepare timesteps
        timesteps = get_timesteps(noise_scheduler, batch_size).to(device=device)


        # 6. Encode input using VAE
        image_latents = encode_vae_image(vae, image, device)
        image_latents = image_latents.to(dtype=dtype)

        # 7. add noise
        latent_model_input = noise_scheduler.add_noise(image_latents, noise, timesteps).to(dtype=dtype).to(device=device)
        latent_model_input = torch.cat([image_latents] * 2) if do_classifier_free_guidance else latent_model_input
        timesteps = torch.cat((timesteps, timesteps), dim=0) if do_classifier_free_guidance else timesteps

    # 8. controlnet
    control_model_input = latent_model_input
    controlnet_prompt_embeds = prompt_embeds
    down_block_res_samples, mid_block_res_sample = controlnet(
        control_model_input,
        timesteps,
        encoder_hidden_states=controlnet_prompt_embeds,
        controlnet_cond=image_c,
        guess_mode=guess_mode,
        return_dict=False,
    )

    # 8. unet
    noise_pred = unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states=prompt_embeds,
        down_block_additional_residuals=down_block_res_samples,
        mid_block_additional_residual=mid_block_res_sample,
        return_dict=False,
    )[0]

    # 9. do_classifier_free_guidance
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    return noise_pred, noise

def prepare_noise(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        vae_scale_factor,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            height // vae_scale_factor,
            width // vae_scale_factor,
        )
        latents = torch.randn(shape, device=device, dtype=dtype)

        return latents

def prepare_image(
        image,
        width,
        height,
        batch_size,
        control_image_processor,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        image = control_image_processor.preprocess(image, height=height, width=width)
        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        image = image.repeat_interleave(repeat_by, dim=0)

        image = image.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            image = torch.cat([image] * 2)

        return image


def get_timesteps(noise_scheduler, batch_size):
    # self.timesteps からランダムにサンプリング
    timesteps_indices = torch.randint(0, len(noise_scheduler.timesteps), (batch_size,))
    timesteps = noise_scheduler.timesteps[timesteps_indices]
    return timesteps

def encode_vae_image(vae, image, device):
    image_latent = vae.encode(image).latent_dist.mode().detach()
    image_latent.to(device)
    return image_latent

def encode_prompt(
        unet,
        text_encoder,
        tokenizer,
        prompt,
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
        """

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            if clip_skip is None:
                prompt_embeds = text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
                prompt_embeds = prompt_embeds[0]
            else:
                prompt_embeds = text_encoder(
                    text_input_ids.to(device), attention_mask=attention_mask, output_hidden_states=True
                )
                # Access the `hidden_states` first, that contains a tuple of
                # all the hidden states from the encoder layers. Then index into
                # the tuple to access the hidden states from the desired layer.
                prompt_embeds = prompt_embeds[-1][-(clip_skip + 1)]
                # We also need to apply the final LayerNorm here to not mess with the
                # representations. The `last_hidden_states` that we typically use for
                # obtaining the final prompt representations passes through the LayerNorm
                # layer.
                prompt_embeds = text_encoder.text_model.final_layer_norm(prompt_embeds)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif unet is not None:
            prompt_embeds_dtype = unet.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)


        return prompt_embeds, negative_prompt_embeds