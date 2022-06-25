from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import io, transforms
from PIL import Image

from data.augmentations import GaussianBlur, Solarization

from config import *

class ImageData(Dataset):
    def __init__(self, files: List[str]):
        self.files = files
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            # transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ])

        self.randcrop_big = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=1.0),
            normalize,
        ])
        self.randcrop_small = transforms.Compose([
            transforms.RandomResizedCrop(256, scale=(0.5, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            normalize,
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i])
        img1 = self.randcrop_big(img)
        img2 = self.randcrop_small(img)
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
        patches = patches.reshape(num_images, NUM_PATCHES, -1)
        return patches
        
    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        x1, x2 = zip(*batch)

        return self.reshape(x1), self.reshape(x2)


class ImageOriginalData(Dataset):
    def __init__(self, files: List[str]):
        self.files = files
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=DATASET_MEAN, std=DATASET_STD),
        ])
        self.resize = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
            normalize
        ])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        img = Image.open(self.files[i])
        return self.resize(img)

class CollateSingleImage(CollateFn):    
    def __call__(
        self, batch: List[torch.Tensor]
    ) -> torch.FloatTensor:
        return self.reshape(batch)