import os
from typing import List

import wandb
import torch
from torchvision import io, transforms
from torchvision.transforms.functional import to_pil_image
import pandas as pd
from PIL import ImageDraw
from config import *

resize = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))

def get_closest(embedding: torch.FloatTensor, i: int):
    similarity = embedding @ embedding[i,:].T # dot product
    scores, idx = similarity.topk(TOPK)
    return scores.cpu().numpy(), idx.cpu().numpy()

def get_closest_wandb_images(embedding: torch.FloatTensor, i: int, files: List[str]):
    df = pd.read_csv(ALL_FILES)
    row = df[df["img_path"].str.contains(os.path.basename(files[i]))]
    main_img = to_pil_image(resize(io.read_image(files[i])))
    closest_imgs = [wandb.Image(main_img)]
    classes = [row["flow_rate"]]
    
    scores, idx = get_closest(embedding, i)
    
    for i, score in zip(idx, scores):
        img = to_pil_image(resize(io.read_image(files[i])))
        row = df[df["img_path"].str.contains(os.path.basename(files[i]))]
        closest_imgs.append(
            wandb.Image(img, caption=f"{score:.4f}")
        )
        classes.append(row["flow_rate"])
    images = [i for i in range(TOPK + 1)]
    data = [[images[i], classes[i]] for i in range(TOPK + 1)]
    table = wandb.Table(data=data, columns=["images", "classes"])
    # closest_imgs.append(
    #    {"classes" : wandb.plot.bar(table, "images", "classes", title="title")}
    # )
    print(len(closest_imgs))
    return closest_imgs