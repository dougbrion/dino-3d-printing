import pandas as pd
import numpy as np
from PIL import Image, ImageFile
from torchvision import transforms
import os
from tqdm import tqdm
import torch

ALL_FILES = "/home/cam/Documents/FastData/dino-test/final.csv"
TRAIN_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
VAL_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
TEST_FILES = "/home/cam/Documents/FastData/dino-test/train.csv"
TRAIN_DIR = "/home/cam/Documents/FastData/"

ImageFile.LOAD_TRUNCATED_IMAGES = True

image_dim = (256, 256)

img_means = None
img_stds = None

psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

mean_lst_1d = []
std_lst_1d = []

df = pd.read_csv(ALL_FILES)

print(len(df))

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    img_path = os.path.join(TRAIN_DIR, row["img_path"])

    img = Image.open(img_path)

    pixels = transforms.ToTensor()(img)
    psum += torch.sum(pixels, dim=[1, 2])
    psum_sq += torch.sum(pixels ** 2, dim=[1, 2])

print(psum)
print(psum_sq)

count = len(df) * image_dim[0] * image_dim[1]

total_mean = psum / count
total_var = (psum_sq / count) - (total_mean ** 2)
total_std = torch.sqrt(total_var)

print(total_mean)
print(total_std)

print("Total dataset")
print("Means for RGB channels:", list(total_mean.numpy()))
print("Variance for RGB channels:", list(total_var.numpy()))
print("Standard deviation for RGB channels:", list(total_std.numpy()))

with open("./data/mean_std.txt", "w+") as f:
    f.write(str(list(total_mean.numpy())))
    f.write("\n")
    f.write(str(list(total_var.numpy())))
    f.write("\n")
    f.write(str(list(total_std.numpy())))
    f.write("\n")
    f.close()