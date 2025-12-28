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
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid

def get_nearest_neighbors_batched(gen_images, dataloader, device):
    """"
    gen_images: Tensor of shape (B, C, 16, 16)
    real_images: Tensor of shape (N, C, 16, 16)
    """
    print("Computing nearest neighbors...")
    B = gen_images.shape[0]
    gen_flat = gen_images.reshape(B, -1).to(device)
    
    # Initialize "Best So Far"
    # We start with infinite distance and empty image placeholders
    closest_dists = torch.full((B,), float('inf'), device=device)
    closest_images = torch.zeros_like(gen_images, device=device)

    for real_batch, _ in dataloader:
        real_batch = real_batch.to(device)
        
        # IMPORTANT: Normalize real images to [0,1] to match generated images
        # Assuming your standard training loop uses [-1, 1], we convert here:
        real_batch_norm = ((real_batch + 1) / 2).clamp(0, 1)
        
        real_flat = real_batch_norm.reshape(real_batch.shape[0], -1)
        
        # Compute distances between (Gen Images) and (Current Real Batch)
        # Shape: (Num_Gen, Num_Real_In_Batch)
        dists = torch.cdist(gen_flat, real_flat, p=2)
        
        # Find the minimum distance in this batch for each generated image
        min_dists_batch, min_indices_batch = torch.min(dists, dim=1)
        
        # Check if we found a new closest match
        better_mask = min_dists_batch < closest_dists
        
        # Update distances
        closest_dists[better_mask] = min_dists_batch[better_mask]
        
        # Update images
        # We find which generated images found a new best friend in this batch
        update_indices = torch.where(better_mask)[0]
        for idx in update_indices:
            # Get the specific image from the real batch
            best_match_idx = min_indices_batch[idx]
            closest_images[idx] = real_batch_norm[best_match_idx]

    return closest_images, closest_dists

def plot_nearest_neighbors(gen_images, real_closest_images, distances, name):
    num_to_check = gen_images.shape[0]
    fig, axes = plt.subplots(num_to_check, 2, figsize=(5, 2.5 * num_to_check))
    
    if num_to_check == 1:
        axes = [axes]

    for i in range(num_to_check):
        # Generated Image
        gen_img_np = gen_images[i].permute(1, 2, 0).cpu().numpy()
        
        # Closest Real Image (Found via streaming)
        real_img_np = real_closest_images[i].permute(1, 2, 0).cpu().numpy()
        dist = distances[i].item()

        # Plot Generated
        axes[i][0].imshow(gen_img_np)
        axes[i][0].set_title("Generated")
        axes[i][0].axis("off")

        # Plot Nearest Real
        axes[i][1].imshow(real_img_np)
        axes[i][1].set_title(f"Nearest Real\n(Dist: {dist:.2f})")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(f"nearest_neighbors_check{name}.png")
    print(f"Saved plot to nearest_neighbors_check{name}.png")


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
    null_class = dataset.labels.max().item() + 1  # Define null class index
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
            # Create a mask where True means "drop this label" (10% probability)
            # shape: (batch_size,)
            dropout_mask = torch.rand(class_labels.shape[0], device=class_labels.device) < 0.1
            training_labels = class_labels.clone()
            training_labels[dropout_mask] = null_class
            # Sample a random timestep for each image
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states = None,  class_labels=training_labels, return_dict=False)[0]
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
                debug_imgs = sample_images.clone().detach()

                print("dtype:", debug_imgs.dtype)
                print("shape:", debug_imgs.shape)
                print("min:", debug_imgs.min().item())
                print("max:", debug_imgs.max().item())
                print("mean:", debug_imgs.mean().item())

                sample_images = ((sample_images.clamp(-1, 1) + 1) * 127.5).round().type(torch.uint8)
                sample_images = sample_images.permute(0, 2, 3, 1).cpu().numpy()
            

            # Save model and create images
            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)
                pipeline.save_pretrained(config.output_dir)
                # save and compare sample images
                model.eval()
                all_fake_images = []

                guidance_scale = 5.0   # How hard to force the label (try 3.0 to 7.0)
                null_class = dataset.labels.max().item() + 1  # The "empty" label used during training dropout

                with torch.no_grad():
                    for _ in range(20):
                        labels = torch.randint(0,
                                                dataset.labels.max().item() + 1,
                                                (config.eval_batch_size_fid,), 
                                                device=device)
                        null_labels = torch.full_like(labels, null_class)
                        combined_labels = torch.cat([null_labels, labels])


                        fake_images = torch.randn((config.eval_batch_size_fid,
                                            3,
                                                config.image_size, 
                                                config.image_size), 
                                                device=device)
                        
                        for t in noise_scheduler.timesteps:
                            # Input shape becomes (2 * Batch_Size, 3, H, W)
                            latent_model_input = torch.cat([fake_images] * 2)
                            noise_pred = model(latent_model_input, 
                                            t, 
                                            class_labels=combined_labels,
                                            return_dict=False,
                                            encoder_hidden_states = None)[0]
                            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                            # 9. Step with the scheduler using the guided noise
                            fake_images = noise_scheduler.step(noise_pred, t, fake_images).prev_sample
                        
                        fake_images = ((fake_images.clamp(-1, 1) + 1) / 2).clamp(0, 1)
                        all_fake_images.append(fake_images.cpu())

                all_fake_images = torch.cat(all_fake_images, dim=0)
                closest_real, closest_dists = get_nearest_neighbors_batched(
                    all_fake_images, 
                    train_dataloader, # Pass the dataloader, not a list!
                    device=model.device
                )
                plot_nearest_neighbors(
                    all_fake_images[:20],
                    closest_real[:20], 
                    closest_dists[:20], 
                    name=f"_epoch{epoch+1}"
                )

                all_fake_images = None
                model.train()
                
            model.train()


if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.cuda.get_device_name(0))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"
    # class counts for calculating class weights
    class_counts = [8000, 32400, 6000, 35000, 8000]
    print("Path to dataset files:", path)
    num_samples = sum(class_counts)
    class_weights = [num_samples / c for c in class_counts]

    

    config = TrainingConfig()
    # load data into dataloader
    dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")

    # add weights to samples
    sample_weights = [class_weights[label] for label in dataset.labels]
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(dataset,
                                   batch_size = config.train_batch_size, 
                                   sampler=sampler,
                                   shuffle = False, 
                                   num_workers=2, 
                                   pin_memory=True)

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
        block_out_channels = (64, 128, 256),
        num_class_embeds=dataset.labels.max().item() + 1 + 1,
        class_embed_type="timestep",
        dropout=0.1,
        mid_block_type = "UNetMidBlock2D",
        down_block_types=
        (
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D"
        ),
        up_block_types=
        (
            "UpBlock2D",
            "AttnUpBlock2D",
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