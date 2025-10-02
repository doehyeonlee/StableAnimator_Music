from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

class MusicEncoder(nn.Module):
    def __init__(self, in_dim=4800, latent_dim=1024):  # cross_attention_dim에 맞춤
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, latent_dim)
        )

    def forward(self, x):  # (B, T, 4800)
        B, T, F = x.shape
        z = self.net(x.view(B*T, F)) 
        z = z.view(B, T, self.latent_dim)  # (B, T, latent_dim)
        return z

    @classmethod
    def from_pretrained(cls, pretrained_model_path, latent_dim=1024):
        """Load pretrained music encoder weights"""
        if not Path(pretrained_model_path).exists():
            raise FileNotFoundError(f"No model file at {pretrained_model_path}")
        
        print(f"Loading MusicEncoder from {pretrained_model_path}")
        
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        
        # Pretrained 모델의 latent_dim 확인
        pretrained_latent_dim = state_dict['net.4.weight'].shape[0]
        
        model = cls(in_dim=4800, latent_dim=pretrained_latent_dim)
        model.load_state_dict(state_dict, strict=True)
        
        # latent_dim이 다르면 projection layer 추가
        if pretrained_latent_dim != latent_dim:
            print(f"Adding projection layer: {pretrained_latent_dim} -> {latent_dim}")
            model.projection = nn.Linear(pretrained_latent_dim, latent_dim)
            model.latent_dim = latent_dim
        
        return model
    
    def forward_with_projection(self, x):
        """Projection layer가 있을 때 사용"""
        z = self.forward(x)  # (B, T, pretrained_latent_dim)
        if hasattr(self, 'projection'):
            z = self.projection(z.view(-1, z.shape[-1]))  # (B*T, latent_dim)
            z = z.view(x.shape[0], x.shape[1], -1)  # (B, T, latent_dim)
        return z
