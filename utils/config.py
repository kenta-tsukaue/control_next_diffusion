from dataclasses import dataclass

@dataclass
class TrainingConfig:
    height = 768  # the generated image resolution
    width = 768
    train_batch_size = 2
    eval_batch_size = 2
    data_path = "/public/tsukaue/datasets/action_youtube_naudio"  #"dataset/action_youtube_naudio"
    num_epochs = 50
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 50
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "output"  # the model name locally and on the HF Hub
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0

