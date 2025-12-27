from dataclasses import dataclass



@dataclass
class TrainingConfig:
    image_size = 16  # the generated image resolution
    train_batch_size = 512  # batch size for training
    eval_batch_size = 16  # how many images to sample during evaluation in training
    eval_batch_size_fid = 32  # how many images to sample during FID evaluation
    eval_size_fid = 50000  # how many images to sample for FID evaluation
    num_epochs = 100
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    save_image_epochs = 10
    save_model_epochs = 10
    mixed_precision = 'fp16'  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = './pixelart_ddpm_version4'  # the model namy locally and on the HF Hub
    deneme_output_dir = './deneme_pixelart_ddpm'  # the model namy locally and on the HF Hub
    push_to_hub = False  # whether to upload the saved model to the HF Hub
    hub_private_repo = False  
    overwrite_output_dir = True  # overwrite the old model when re-running the notebook
    seed = 0
