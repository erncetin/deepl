from diffusers import UNet2DConditionModel, DDPMPipeline, DDPMScheduler
import torch
from config_file import TrainingConfig
from CustomDataset import CustomDateset
from PIL import Image
import os
from tqdm.auto import tqdm

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid



config = TrainingConfig()

path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"
dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")

model = UNet2DConditionModel.from_pretrained("C:/Users/HP/Desktop/VSCODE_FILES/deep_learning_project/pixelart_ddpm_version2/unet", use_safetensors=True)

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sample_labels = torch.randint(0, dataset.labels.max().item() + 1, (config.eval_batch_size,), device=device)
sample_images = torch.randn((config.eval_batch_size, 3, config.image_size, config.image_size), device=device)

with torch.no_grad():
    for t in noise_scheduler.timesteps:
        noise_pred = model(sample_images, t, class_labels=sample_labels, return_dict=False, encoder_hidden_states = None)[0]
        step = noise_scheduler.step(noise_pred, t, sample_images)
        sample_images = step.prev_sample

    # Convert to PIL and save
    sample_images = ((sample_images.clamp(-1, 1) + 1) * 127.5).byte()
    sample_images = sample_images.permute(0, 2, 3, 1).cpu().numpy()

images = [Image.fromarray(im) for im in sample_images]
grid = make_grid(images, rows=4, cols=4)
os.makedirs(f"{config.deneme_output_dir}/samples", exist_ok=True)
grid.save(f"{config.deneme_output_dir}/samples/deneme.png")