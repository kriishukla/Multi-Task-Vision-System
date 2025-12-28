import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class CamVidDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, label_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.label_transform = label_transform
        self.images = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
        self.labels = sorted([f for f in os.listdir(label_dir) if f.endswith(".png")])
        assert len(self.images) == len(self.labels), "Mismatch between image and label count"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        label_path = os.path.join(self.label_dir, self.labels[idx])
        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB") 
        if self.transform:
            print(2)
            image = self.transform(image)
        if self.label_transform:
            print(2)
            label = self.label_transform(label)
        else:
            
            label = torch.as_tensor(np.array(label), dtype=torch.long)
        return image, label
