import os

import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import wandb

from config import *
from data.dataset import ImageData, ImageOriginalData, CollateFn, CollateSingleImage
from model.loss import HLoss
from model.transformer import Model
from model.training import LightningModel

wandb.login(key=os.environ.get("WANDB_API_KEY"))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(torch.__version__, pl.__version__, wandb.__version__)

files = pd.read_csv(TRAIN_FILES)["img_path"].tolist()
files = [os.path.join(TRAIN_DIR, path) for path in files]
# files = files[:1000]
print("Number of images:", len(files))

train_files, valid_files = train_test_split(files, test_size=0.15, random_state=42)
print(len(train_files), len(valid_files))

train_data = ImageData(train_files)
train_dl = DataLoader(
    train_data, 
    BATCH_SIZE, 
    shuffle=True, 
    drop_last=True, 
    num_workers=4,
    pin_memory=True,
    collate_fn=CollateFn(),
)

valid_data = ImageOriginalData(valid_files)
valid_dl = DataLoader(
    valid_data, 
    BATCH_SIZE*2, 
    shuffle=False, 
    drop_last=False, 
    num_workers=4,
    pin_memory=True,
    collate_fn=CollateSingleImage(),
)

teacher = Model(NUM_PIXELS, N_HEADS, N_LAYERS)
h_loss = HLoss(TEMPERATURE_T, TEMPERATURE_S)
lightning_model = LightningModel(
    teacher, 
    LR,
    h_loss,
    valid_files,
    NUM_PIXELS,
    CENTER_MOMENTUM, 
    TEACHER_MOMENTUM,
    max_epochs=EPOCHS,
)

logger = WandbLogger("DINO", "logs/", project="DINO")
trainer = pl.Trainer(
    max_epochs=EPOCHS,
    accelerator="gpu",
    devices=torch.cuda.device_count(),
    strategy="ddp_find_unused_parameters_false",
    gradient_clip_val=1.0,
    logger=logger,
    precision=16,
    num_sanity_val_steps=0,
)
print("Training...")
trainer.fit(lightning_model, train_dl, valid_dl)