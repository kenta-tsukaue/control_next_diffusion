from dataclasses import dataclass

@dataclass
class TrainingConfig:
    height = 768  # the generated image resolution
    width = 768
    train_batch_size = 2
    eval_batch_size = 2
    data_path = "/public/tsukaue/datasets/action_youtube_naudio"  #"dataset/action_youtube_naudio"
    num_epochs = 100
    gradient_accumulation_steps = 8
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 5
    save_model_epochs = 20
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_weight_decay = 1e-2,
    adam_epsilon=1e-08


