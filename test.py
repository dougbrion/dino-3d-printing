
import os

import torch
import pandas as pd
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from model.transformer import Model
from config import *
from data.dataset import ImageOriginalData, CollateSingleImage

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

files = pd.read_csv(TRAIN_FILES)["img_path"].tolist()
files = [os.path.join(TRAIN_DIR, path) for path in files]

image_orig_data = ImageOriginalData(files)
image_orig_dl = DataLoader(
    image_orig_data, 
    BATCH_SIZE*2, 
    shuffle=False, 
    drop_last=False, 
    num_workers=4,
    pin_memory=True,
    collate_fn=CollateSingleImage(),
)

checkpoint_path = "logs/DINO/21spboj3/checkpoints/epoch=4-step=13285.ckpt"

teacher = Model(NUM_PIXELS, N_HEADS, N_LAYERS).load_state_dict(
    torch.load(checkpoint_path)["state_dict"]["teacher"]
)

teacher = teacher.eval().to(device)
embedding = []
with torch.no_grad():
    for x in tqdm(image_orig_dl):
        out = teacher(x.to(device))
        embedding.append(out.cpu())
        
    embedding = torch.cat(embedding, dim=0)