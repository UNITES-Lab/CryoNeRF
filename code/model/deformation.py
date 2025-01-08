import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torchvision.models import resnet18


class ResLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) + x


class DeformationEncoder(nn.Module):
    def __init__(self, encoder_type: str, latent_dim: int, checkpointing=False, size=256, hartley=False) -> None:
        super().__init__()
        
        # self.checkpointing = checkpointing
        
        self.encoder = timm.create_model(encoder_type, num_classes=0, global_pool="", in_chans=1)
        self.output_layer = nn.Sequential(nn.LazyLinear(128), nn.ReLU(), nn.Linear(128, latent_dim))
        # self.normalize = nn.BatchNorm2d(num_features=1)

        # dummy input for initialization
        if hartley:
            self.output_layer(self.encoder(torch.randn(1, 1, size + 1, size + 1)).reshape(1, -1))
        else:
            self.output_layer(self.encoder(torch.randn(1, 1, size, size)).reshape(1, -1))
            
    def forward(self, images):
        x = F.relu(self.encoder(images).reshape(images.shape[0], -1))
        latent_variable = self.output_layer(x)
        # latent_variable = self.encoder(rearrange(images, "B N H W -> B (N H W)"))
        
        return latent_variable
    
    
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.fc1 = nn.Linear(latent_dim, 128 * 16 * 16)
        
        self.conv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        
        x = x.view(-1, 128, 16, 16)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        
        return x
    

class DeformationDecoder(nn.Module):
    def __init__(self, dim_after_enc, latent_dim=16, hid_dim=128, hid_layer_num=2, checkpointing=False) -> None:
        super().__init__()
        
        # self.active_func = nn.LeakyReLU
        self.active_func = nn.ReLU
        self.checkpointing = checkpointing
        self.hid_layer_num = hid_layer_num
        self.module_list = [nn.Linear(dim_after_enc + latent_dim, hid_dim), self.active_func()]
        # for i in range(hid_layer_num):
        #     self.module_list += [nn.Linear(hid_dim, hid_dim), self.active_func()]
        for i in range(hid_layer_num):
            self.module_list += [ResLinear(hid_dim, hid_dim), self.active_func()]
        self.module_list += [nn.Linear(hid_dim, hid_dim // 2), self.active_func(), nn.Linear(hid_dim // 2, 3)]
        
        self.mlp = nn.Sequential(*self.module_list)
        
    def forward(self, encoded_pos, latent_variable):
        encoded_pos_with_latent = torch.cat([encoded_pos, repeat(latent_variable, "B Dim -> B N L Dim",
                                                                 N=encoded_pos.shape[1], L=encoded_pos.shape[2])], dim=-1)
        
        if self.checkpointing:
            delta_coord = checkpoint_sequential(self.mlp, len(self.mlp), encoded_pos_with_latent)
        else:
            delta_coord = self.mlp(encoded_pos_with_latent)

        return delta_coord

class DeformantionEmbedding(nn.Module):
    def __init__(self, length, dfom_latent_dim=16):
        super().__init__()

        self.embeddings = nn.Embedding(length, dfom_latent_dim)
        nn.init.kaiming_uniform_(self.embeddings.weight)

    def forward(self, indices):
        return self.embeddings(indices)
