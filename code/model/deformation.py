import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    def __init__(self, encoder_type: str, latent_dim: int, size=256, hartley=False) -> None:
        super().__init__()
        
        self.encoder = timm.create_model(encoder_type, num_classes=0, global_pool="", in_chans=1)
        self.output_layer = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, latent_dim))

        # dummy input for initialization
        if hartley:
            self.output_layer(self.encoder(torch.randn(1, 1, size + 1, size + 1)).reshape(1, -1))
        else:
            self.output_layer(self.encoder(torch.randn(1, 1, size, size)).reshape(1, -1))
            
    def forward(self, images):
        x = F.relu(self.encoder(images).reshape(images.shape[0], -1))
        latent_variable = self.output_layer(x)
        
        return latent_variable
