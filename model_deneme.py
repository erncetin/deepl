from diffusers import UNet2DConditionModel, DDIMScheduler, DDPMScheduler
import torch
from config_file import TrainingConfig
from CustomDataset import CustomDateset
from PIL import Image
import os
from tqdm.auto import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import DataLoader
from torch.utils.data import Subset
if __name__ == "__main__":
    config = TrainingConfig()
    path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"

    dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")
    num_real = 50000
    idx = torch.randperm(len(dataset))[:num_real]
    fid_dataset = Subset(dataset, idx)
    real_dataloader = DataLoader(fid_dataset, batch_size = 256, shuffle = False, num_workers=2, pin_memory=True)
    

    model = UNet2DConditionModel.from_pretrained("C:/Users/HP/Desktop/VSCODE_FILES/deep_learning_project/pixelart_ddpm_version4/unet", use_safetensors=True)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise_scheduler = DDIMScheduler.from_config(noise_scheduler.config)
    noise_scheduler.set_timesteps(50)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(device)


    fid = FrechetInceptionDistance(normalize=True).to(device)
    model.eval()
    print("FID device:", fid.device)
    print("Model device:", next(model.parameters()).device)
    with torch.no_grad():
        for real_images, _ in tqdm(real_dataloader, desc="FID real"):
            real_images = real_images.to(device)
            real_images = ((real_images + 1) / 2).clamp(0, 1)
            fid.update(real_images, real=True)

    print(real_images.shape)
    print(real_images.dtype)
    print(real_images.min().item(), real_images.max().item())
    with torch.no_grad():
        for _ in range(config.eval_size_fid // config.eval_batch_size_fid):
            labels = torch.randint(0,
                                    dataset.labels.max().item() + 1,
                                    (config.eval_batch_size_fid,), 
                                    device=device)
            images = torch.randn((config.eval_batch_size_fid,
                                3,
                                    config.image_size, 
                                    config.image_size), 
                                    device=device)
            
            for t in noise_scheduler.timesteps:
                noise_pred = model(images, 
                                t, 
                                class_labels=labels,
                                return_dict=False,
                                encoder_hidden_states = None)[0]
                images = noise_scheduler.step(noise_pred,t,images).prev_sample
            
            images = ((images.clamp(-1, 1) + 1) * 127.5).byte()
            fid.update(images, real=False)

    fid_score = fid.compute()
    print("FID score:", fid_score.item())
