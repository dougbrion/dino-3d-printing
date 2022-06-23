from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import io, transforms
from PIL import Image

from config import *

class ImageData(Dataset):
    def __init__(self, files: List[str]):
        self.files = files
        self.randcrop_big = transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE), scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ])
        self.randcrop_small = transforms.Compose([
            transforms.RandomResizedCrop((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i])
        img1 = self.randcrop_big(img)
        img2 = self.randcrop_small(img)
        if img.size[0] == 1:
            img1 = torch.cat([img1]*3)
            img2 = torch.cat([img2]*3)

        return img1, img2

class CollateFn:
    def reshape(self, batch):
        patches = torch.stack(batch)
        patches = patches.unfold(2, PATCH_SIZE, PATCH_SIZE)
        patches = patches.unfold(3, PATCH_SIZE, PATCH_SIZE)

        num_images = len(patches)
        patches = patches.reshape(
            num_images,
            RGB_CHANNELS, 
            NUM_PATCHES, 
            PATCH_SIZE, 
            PATCH_SIZE
        )
        patches.transpose_(1, 2)
        patches = patches.reshape(num_images, NUM_PATCHES, -1) / 255.0 - 0.5
        return patches
        
    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        x1, x2 = zip(*batch)

        return self.reshape(x1), self.reshape(x2)


class ImageOriginalData(Dataset):
    def __init__(self, files: List[str]):
        self.files = files
        self.resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = io.read_image(self.files[i])
        if img.shape[0] == 1:
            img = torch.cat([img]*3)

        return self.resize(img)

class CollateSingleImage(CollateFn):    
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> torch.FloatTensor:
        return self.reshape(batch)