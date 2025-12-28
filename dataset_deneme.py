from CustomDataset import CustomDateset
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from config_file import TrainingConfig
import torchvision
config = TrainingConfig()
# 1. class =  hayvanlar = zor öğreniyo

# 3. class = eşyalar,silahlar = kolay öğrendikleri
# 4. class = silahlı npcler = orta zorluk?
def imshow(img):
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0))) # CHW to HWC
        plt.show()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    path = "C:/Users/HP/.cache/kagglehub/datasets/ebrahimelgazar/pixel-art/versions/1"

    dataset = CustomDateset(path + "/sprites.npy", path + "/sprites_labels.npy")
    dataloader = DataLoader(dataset, batch_size = 1000, shuffle = True, num_workers=2, pin_memory=True)

    target_class = 4
    x, y = next(iter(dataloader))
    mask = (y == target_class)
    x_class = x[mask]
    print(f"Found {x_class.size(0)} samples of class {target_class}")
    
    if x_class.size(0) > 0:
        imshow(torchvision.utils.make_grid(x_class))