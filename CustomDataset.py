from  torch.utils.data import Dataset
import numpy as np
import torchvision.transforms as T
from PIL import Image
import torch


class CustomDateset(Dataset):
    def __init__(self, npy_path, labels_path, image_size = 16, augment = True):

        data = np.load(npy_path).astype(np.float32)

        self.x = data

        raw_labels = np.load(labels_path)

        if raw_labels.ndim == 2:
            # Convert one-hot encoded labels to class indices
            self.labels = np.argmax(raw_labels, axis=1)
        else:
            # labels are already in class index format
            self.labels = raw_labels

        self.image_size = image_size
        self.augment = augment
        self.transform = T.Compose([
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3),
        ])

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        image = self.x[idx]
        label = self.labels[idx]
        
        # Convert image to tensor, normalize, and add channel dimension
        image = torch.from_numpy(image).float() / 255.0  # Normalize to [0, 1]

        # No need to unsqueeze for RGB images:
        image = image.permute(2, 0, 1) # Change from HxWx3 to 3xHxW (CHW)

        if self.transform:
            image = self.transform(image)


        return image, torch.tensor(label, dtype=torch.long)