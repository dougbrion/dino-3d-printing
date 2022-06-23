from typing import Callable, List, Tuple, Any, Dict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from deepspeed.ops.adam import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR

import wandb

from data.helpers import get_closest_wandb_images
from config import *

class LightningModel(pl.LightningModule):
    def __init__(
        self,
        teacher: nn.Module,
        lr: float,
        loss_fn: Callable,
        valid_files: List[str],
        dim: int,
        center_momentum: float,
        param_momentum: float,
        max_epochs: int,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = copy.deepcopy(teacher)
        self.lr = lr
        self.loss_fn = loss_fn
        self.c_mom = center_momentum
        self.p_mom = param_momentum
        self.register_buffer("center", torch.zeros((1, dim)).float())
        self.valid_files = valid_files
        self.max_epochs = max_epochs
        
        for p in self.teacher.parameters():
            p.requires_grad = False

    def loss_calculation(
        self,
        batch: Tuple[torch.FloatTensor, torch.FloatTensor],
    ) -> torch.FloatTensor:
        x1, x2 = batch
        
        s1, s2 = self.student(x1), self.student(x2)
        t1, t2 = self.teacher(x1), self.teacher(x2)
        
        loss = self.loss_fn(t1, s2, self.center) + self.loss_fn(t2, s1, self.center)
        
        emperical_center = F.normalize(
            torch.cat([t1, t2]).mean(dim=0, keepdims=True),
            dim=-1,
        )

        return loss, emperical_center

    def training_step(
        self, batch: Tuple[torch.FloatTensor, torch.FloatTensor], *args: List[Any]
    ) -> torch.Tensor:
        
        loss, emperical_center = self.loss_calculation(batch)
        self.log(name="Training loss", value=loss, on_step=True, on_epoch=True)
        
        self.center = F.normalize(
            self.c_mom * self.center + (1 - self.c_mom) * emperical_center,
            dim=-1,
        )
        for s_p, t_p in zip(self.student.parameters(), self.teacher.parameters()):
            t_p.data = self.p_mom * t_p.data + (1 - self.p_mom) * s_p.data
            
        return loss
    
    def validation_step(self, images: torch.FloatTensor, *args: List[Any]) -> None:
        return self.teacher(images)
        
    def validation_epoch_end(self, validation_step_outputs):
        valid_embeds = torch.cat([pred for pred in validation_step_outputs])
        columns = ["image"] + [f"closest_{i+1}" for i in range(TOPK)]
        indices = np.random.choice(len(valid_embeds), VALID_IMAGES, replace=False)
        rows = [get_closest_wandb_images(valid_embeds, i, self.valid_files) for i in indices]
        table = wandb.Table(data=rows, columns=columns)
        self.logger.experiment.log({f"epoch {self.current_epoch} results": table})
        
    def on_after_backward(self):
        if self.trainer.global_step % 50 == 0:  # don't make the tf file huge
            global_step = self.trainer.global_step
            for name, param in self.student.named_parameters():
                if "weight" in name and not "norm" in name and param.requires_grad:
                    self.logger.experiment.log(
                        {f"{name}_grad": wandb.Histogram(param.grad.cpu())}
                    )

    def configure_optimizers(self) -> Dict:
        optimizer = FusedAdam(self.student.parameters(), lr=self.lr)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=0, last_epoch=-1
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}