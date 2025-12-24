import kagglehub
from CustomDataset import CustomDateset
from torch.utils.data import DataLoader
from diffusers import UNet2DConditionModel, DDPMScheduler
from PIL import Image
import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from config_file import TrainingConfig
import os
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusers import DDPMPipeline

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    logging_dir = os.path.join(config.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=config.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps, 
        project_config=accelerator_project_config,
    )
    if accelerator.is_main_process and config.output_dir is not None:
        os.makedirs(config.output_dir, exist_ok=True)
        accelerator.init_trackers("train_example")
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the 
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch[0]
            class_labels = batch[1]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Move to device
            clean_images = clean_images.to(model.device)
            class_labels = class_labels.to(model.device)

            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states = None,  class_labels=class_labels, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

         # After each epoch, optionally sample demo images and save model
        if accelerator.is_main_process:
            model.eval()
            with torch.no_grad():
                sample_labels = torch.randint(0, dataset.labels.max().item() + 1, (config.eval_batch_size,), device=model.device)
                sample_images = torch.randn((config.eval_batch_size, 3, config.image_size, config.image_size), device=model.device)


                # DDPM reverse process
                for t in noise_scheduler.timesteps:
                        noise_pred = model(sample_images, t, class_labels=sample_labels, return_dict=False, encoder_hidden_states = None)[0]
                        step = noise_scheduler.step(noise_pred, t, sample_images)
                        sample_images = step.prev_sample

                # Convert to PIL and save
                sample_images = ((sample_images.clamp(-1, 1) + 1) * 127.5).byte()
                sample_images = sample_images.permute(0, 2, 3, 1).cpu().numpy()
            images = [Image.fromarray(im) for im in sample_images]
            grid = make_grid(images, rows=4, cols=4)
            os.makedirs(f"{config.output_dir}/samples", exist_ok=True)
            grid.save(f"{config.output_dir}/samples/epoch_{epoch:04d}.png")

            # Save model
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(config.output_dir)
            model.train()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))

    path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"

    print("Path to dataset files:", path)

    config = TrainingConfig()
    # load data into dataloader
    dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")
    train_dataloader = DataLoader(dataset, batch_size = config.train_batch_size, shuffle = True, num_workers=2, pin_memory=True)

    # example data batch
    for imgs, labels in train_dataloader:
        print("Batch of images shape:", imgs.shape)
        print("Batch of labels shape:", labels.shape)
        break
    print(dataset.labels.max().item() + 1)

    # model architecture
    model = UNet2DConditionModel(
        sample_size=config.image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels = (32, 64, 128, 128),
        num_class_embeds=dataset.labels.max().item() + 1,
        class_embed_type="timestep",
        mid_block_type = "UNetMidBlock2D",
        down_block_types=
        (
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=
        (
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D"
        )

    )
    
    # sample images shapes
    sample_image = dataset[0][0].unsqueeze(0)
    print('Input shape:', sample_image.shape)
    print('Output shape:', model(sample_image, timestep=0, encoder_hidden_states = None, class_labels = dataset[0][1].unsqueeze(0)).sample.shape)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

    # show the image 
    noise = torch.randn(sample_image.shape)
    timeeteps = torch.tensor([50])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timeeteps)
    noisy_pil = Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])
    noisy_pil.show()

    #optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    #learning rate scheduler
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * config.num_epochs)
    )
    # training loop
    train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler)
    print("Training completed.")