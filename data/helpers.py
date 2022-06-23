from typing import List

import wandb
import torch
from torchvision import io, transforms
from torchvision.transforms.functional import to_pil_image

from config import *

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

def get_closest(embedding: torch.FloatTensor, i: int):
    similarity = embedding @ embedding[i,:].T # dot product
    scores, idx = similarity.topk(TOPK)
    return scores.cpu().numpy(), idx.cpu().numpy()

def get_closest_wandb_images(embedding: torch.FloatTensor, i: int, files: List[str]):
    main_img = to_pil_image(resize(io.read_image(files[i])))
    closest_imgs = [wandb.Image(main_img)]
    
    scores, idx = get_closest(embedding, i)
    
    for i, score in zip(idx, scores):
        img = to_pil_image(resize(io.read_image(files[i])))
        closest_imgs.append(
            wandb.Image(img, caption=f"{score:.4f}")
        )
        
    return closest_imgs