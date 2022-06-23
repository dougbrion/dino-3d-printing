import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *

class Model(nn.Module):
    def __init__(self, d_model, n_head, n_layers):
        super().__init__()
        # transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # positional embedding
        w_pos = torch.randn(NUM_PATCHES, d_model) / d_model ** 0.5
        cls_token = torch.randn(1, d_model) / d_model ** 0.5
        self.register_parameter("pos_embed", nn.Parameter(w_pos))
        self.register_parameter("cls_token", nn.Parameter(cls_token))

        # pixel projection
        self.linear = nn.Linear(2 * d_model, d_model)
        self.norm1 = nn.LayerNorm(2 * d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        batch_size = len(x)
        position = torch.stack([self.pos_embed] * batch_size)
        x = torch.cat([x, position], dim=-1)
        pixel_proj = self.norm2(F.relu(self.linear(self.norm1(x))))
        batched_cls_token = torch.stack([self.cls_token]*batch_size)
        cls_x = torch.cat([batched_cls_token, pixel_proj], dim=1)
        
        cls_x.transpose_(0, 1)
        return F.normalize(self.encoder(cls_x)[0, ...], dim=-1)