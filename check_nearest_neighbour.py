from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import torch
from config_file import TrainingConfig
from CustomDataset import CustomDateset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
config = TrainingConfig()



def check_nearest_neighbors(gen_images, real_images, num_to_check = 5):
    """"
    gen_images: Tensor of shape (B, C, 16, 16)
    real_images: Tensor of shape (N, C, 16, 16)
    """
    print("Computing nearest neighbors...")
    gen_flat = gen_images.reshape(gen_images.shape[0], - 1).cpu()  # (B, C*16*16)
    real_flat = real_images.reshape(real_images.shape[0], -1).cpu()  # (N, C*16*16)

    if gen_flat.shape[0] > num_to_check:
        gen_flat = gen_flat[:num_to_check]
        gen_images = gen_images[:num_to_check]

    # 3. Compute pairwise Euclidean distance
    # shape: (num_to_check, N_real_images)
    dists = torch.cdist(gen_flat, real_flat, p=2)

    # 4. Find the index of the minimum distance for each generated image
    min_dists, min_indices = torch.min(dists, dim=1)

    # 5. Plotting
    fig, axes = plt.subplots(num_to_check, 2, figsize=(5, 2.5 * num_to_check))
    
    # Handle case where num_to_check is 1
    if num_to_check == 1:
        axes = [axes]

    for i in range(num_to_check):
        # Get the generated image
        gen_img_np = gen_images[i].permute(1, 2, 0).cpu().numpy()
        
        # Get the closest real image
        closest_real_idx = min_indices[i].item()
        real_img_np = real_images[closest_real_idx].permute(1, 2, 0).cpu().numpy()

        # Plot Generated
        axes[i][0].imshow(gen_img_np)
        axes[i][0].set_title("Generated")
        axes[i][0].axis("off")

        # Plot Nearest Real
        axes[i][1].imshow(real_img_np)
        axes[i][1].set_title(f"Nearest Real\n(Dist: {min_dists[i]:.2f})")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig("nearest_neighbors_check.png")
    plt.show()
    print("Saved plot to nearest_neighbors_check.png")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"
    model = UNet2DConditionModel.from_pretrained("C:/Users/HP/Desktop/VSCODE_FILES/deep_learning_project/pixelart_ddpm_version4/unet", use_safetensors=True)
    model.to(device)
    dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")
    dataloader = DataLoader(dataset, batch_size = config.train_batch_size, shuffle = True, num_workers=2, pin_memory=True)
    

    # real images
    all_real_images = []
    with torch.no_grad():
        for real_images, _ in tqdm(dataloader, desc="FID real"):
            real_images = real_images.to(device)
            real_images = ((real_images + 1) / 2).clamp(0, 1)
            all_real_images.append(real_images.cpu())

    all_real_images = torch.cat(all_real_images, dim = 0)  # (N, C, H, W)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    noise_scheduler.set_timesteps(25)

    # fake images
    all_fake_images = []
    with torch.no_grad():
        for _ in range(5):
            labels = torch.randint(0,
                                    dataset.labels.max().item() + 1,
                                    (config.eval_batch_size_fid,), 
                                    device=device)
            fake_images = torch.randn((config.eval_batch_size_fid,
                                3,
                                    config.image_size, 
                                    config.image_size), 
                                    device=device)
            
            for t in noise_scheduler.timesteps:
                noise_pred = model(fake_images, 
                                t, 
                                class_labels=labels,
                                return_dict=False,
                                encoder_hidden_states = None)[0]
                fake_images = noise_scheduler.step(noise_pred,t,fake_images).prev_sample
            
            fake_images = ((fake_images.clamp(-1, 1) + 1) / 2).clamp(0, 1)
            all_fake_images.append(fake_images.cpu())

    all_fake_images = torch.cat(all_fake_images, dim=0)
    check_nearest_neighbors(all_fake_images, all_real_images, num_to_check=5)