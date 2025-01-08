import torch
import torch.nn as nn
from einops import rearrange, repeat
from torch.utils.checkpoint import checkpoint_sequential

class ResLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x) + x


class NeuralRadianceField(nn.Module):
    def __init__(self, dim_after_enc, latent_dim=0, hid_dim=128, hid_layer_num=2, checkpointing=False):
        super().__init__()
        
        self.active_func = nn.ReLU
        self.checkpointing = checkpointing
        self.hid_layer_num = hid_layer_num
        self.module_list = [nn.LazyLinear(hid_dim), self.active_func()]
        # self.module_list = [nn.Linear(dim_after_enc, hid_dim), self.active_func()]
        # for i in range(hid_layer_num):
        #     self.module_list += [nn.Linear(hid_dim, hid_dim), self.active_func()]
        for i in range(hid_layer_num):
            self.module_list += [ResLinear(hid_dim, hid_dim), self.active_func()]
        self.module_list += [nn.Linear(hid_dim, 1)]
        
        self.mlp = nn.Sequential(*self.module_list)

    def forward(self, encoded_pos, latent_variable=None):
        if latent_variable is not None:
            encoded_pos_with_latent = torch.cat([encoded_pos, repeat(latent_variable, "B Dim -> B N L Dim",
                                                                     N=encoded_pos.shape[1], L=encoded_pos.shape[2])], dim=-1)
        else:
            encoded_pos_with_latent = encoded_pos
        # encoded_pos_with_latent = encoded_pos
        if self.checkpointing:
            density = checkpoint_sequential(self.mlp, len(self.mlp), encoded_pos_with_latent)
        else:
            density = self.mlp(encoded_pos_with_latent)
            
        return density