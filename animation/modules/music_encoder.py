from pathlib import Path

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from diffusers.models.modeling_utils import ModelMixin

class MusicEncoder(ModelMixin):
    def __init__(self, indim=4800, hw=64, noise_latent_channels=320):
        super().__init__()
        self.hw = hw
        self.noise_latent_channels = noise_latent_channels
        self.latent_dim = hw * hw

        # projection to 64x64
        self.net = nn.Linear(indim, self.latent_dim)

        # 1→noise_latent_channels 채널 확장
        self.expand = nn.Conv2d(1, noise_latent_channels, kernel_size=3, padding=1)

    def forward(self, x):  # (B, T, 4800)
        B, T, F = x.shape
        z = self.net(x.view(B*T, F))                      # (B*T, 4096)
        z = z.view(B*T, 1, self.hw, self.hw)              # (B*T, 1, 64, 64)
        z = self.expand(z)                                # (B*T, noise_latent_channels, 64, 64)
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
