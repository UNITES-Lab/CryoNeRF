import numpy as np
import tinycudann as tcnn
import torch
import torch.nn as nn
from torch.fft import fft2, fftn, fftshift, ifftn, ifftshift


def to_cuda(sample: dict):
    cuda_sample = {}
    for k, v in sample.items():
        if type(v) == torch.Tensor:
            cuda_sample[k] = v.cuda()
        else:
            cuda_sample[k] = v

    return cuda_sample


def positional_encoding(coords: torch.Tensor, enc_dim=10):
    '''
    coords (B, N, 3): The 3D coordinates of position to query.
    '''
    # print(coords.requires_grad)
    freqs = torch.exp2(torch.arange(enc_dim)) / 10
    freqs = freqs.repeat_interleave(2).reshape(1, 1, 1, 1, 2 * enc_dim).to(coords.device)  # (1, 1, 1, 1, 2L)

    coords_time_freqs = coords.unsqueeze(-1) * freqs  # (B, N, S, 3, 1) * (1, 1, 1, 1, 2L) -> (B, N, S, 3, 2L)

    sin_part = torch.sin(coords_time_freqs[..., 0::2])
    cos_part = torch.cos(coords_time_freqs[..., 1::2])
    encoded_pos = torch.cat([sin_part.unsqueeze(-1), cos_part.unsqueeze(-1)], dim=-1)

    return encoded_pos.reshape((*coords_time_freqs.shape[:3], -1))


class positional_encoding_nerf():
    def __init__(self, size, enc_dim=10) -> None:
        self.size = size
        self.enc_dim = enc_dim
    
    def get_dim(self):
        return self.enc_dim * 6, self
    
    def __call__(self, coords):
        '''
        coords (B, N, 3): The 3D coordinates of position to query.
        '''
        print(coords.shape)
        freqs = torch.exp2(torch.arange(self.enc_dim)) / 10
        freqs = freqs.repeat_interleave(2).reshape(1, 1, 1, 1, 2 * self.enc_dim).to(coords.device)  # (1, 1, 1, 1, 2L)

        coords_time_freqs = coords.unsqueeze(-1) * freqs  # (B, N, S, 3, 1) * (1, 1, 1, 1, 2L) -> (B, N, S, 3, 2L)

        sin_part = torch.sin(coords_time_freqs[..., 0::2])
        cos_part = torch.cos(coords_time_freqs[..., 1::2])
        encoded_pos = torch.cat([sin_part.unsqueeze(-1), cos_part.unsqueeze(-1)], dim=-1)

        return encoded_pos.reshape((*coords_time_freqs.shape[:3], -1))


class positional_encoding_geom():
    def __init__(self, size, enc_dim=10):
        self.size = size
        self.enc_dim = enc_dim
        
    def get_dim(self):
        return self.enc_dim * 6, self
        
    def __call__(self, coords):
        """Expand coordinates in the Fourier basis with geometrically spaced wavelengths from 2/D to 2pi"""
        freqs = torch.arange(self.enc_dim, dtype=torch.float, device=coords.device)
        freqs = (2 * np.pi * (self.size / 2) ** (freqs / (self.enc_dim - 1)))  # option 1: 2/D to 1
        freqs = freqs.view(*[1] * len(coords.shape), -1)  # 1 x 1 x D2
        coords = coords.unsqueeze(-1)  # B x 3 x 1
        x = torch.cat([torch.sin(coords * freqs), torch.cos(coords * freqs)], -1)  # B x 3 x D
        x = x.view(*coords.shape[:-2], self.enc_dim * 6)  # B x in_dim-zdim
        
        return x


class positional_encoding_grid(nn.Module):
    def __init__(self, size, enc_dim=10) -> None:
        super().__init__()
        
        self.config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.4472692012786865,
        }
        
        self.encoding = tcnn.Encoding(3, self.config, dtype=torch.float32)
        
    def get_dim(self):
        return self.config["n_levels"] * self.config["n_features_per_level"], self
    
    def forward(self, coords):
        orig_shape = coords.shape
        encoded_pos = self.encoding(coords.reshape(-1, 3)).reshape((*orig_shape[:-1], -1))
        
        return encoded_pos
    
    
class positional_encoding_gaussian(nn.Module):
    def __init__(self, size, enc_dim=10) -> None:
        self.size = size
        self.enc_dim = enc_dim
        self.rand_freqs = nn.Parameter(torch.randn((3 * self.enc_dim, 3), dtype=torch.float) * 0.5, requires_grad=False)
    
    def get_dim(self):
        return self.enc_dim * 6, self
    
    def forward(self, coords):
        assert self.rand_freqs is not None
        # k = coords . rand_freqs
        # expand rand_freqs with singleton dimension along the batch dimensions
        # e.g. dim (1, ..., 1, n_rand_feats, 3)
        freqs = self.rand_freqs.view(*[1] * (len(coords.shape) - 1), -1, 3) * self.D2

        kxkykz = coords[..., None, 0:3] * freqs  # compute the x,y,z components of k
        k = kxkykz.sum(-1)  # compute k
        s = torch.sin(k)
        c = torch.cos(k)
        x = torch.cat([s, c], -1)
        x = x.view(*coords.shape[:-1], self.in_dim - self.zdim)
        if self.zdim > 0:
            x = torch.cat([x, coords[..., 3:]], -1)
            assert x.shape[-1] == self.in_dim
        print(x.shape)
        return x


def draw_inscribed_sphere(n):
    volume = np.zeros((n, n, n), dtype=np.float32)
    
    center = (n / 2, n / 2, n / 2)
    
    radius = n / 2

    z, y, x = np.indices((n, n, n))
    distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    
    volume[distance_from_center <= radius] = 1
    
    return volume

def window_mask(D, in_rad: float, out_rad: float) -> torch.Tensor:
    """
    Create a square radial mask of linearly-interpolated float values
    from 1.0 (within in_rad of center) to 0.0 (beyond out_rad of center)
    Args:
        D: Side length of the (square) mask
        in_rad: inner radius (fractional float between 0 and 1) inside which all values are 1.0
        out_rad: outer radius (fractional float between 0 and 1) beyond which all values are 0.0

    Returns:
        A 2D Tensor of shape (D, D) of mask values between 0 (inclusive) and 1 (inclusive)
    """
    assert D % 2 == 0
    assert in_rad <= out_rad
    x0, x1 = torch.meshgrid(
        torch.linspace(-1, 1, D + 1, dtype=torch.float32)[:-1],
        torch.linspace(-1, 1, D + 1, dtype=torch.float32)[:-1],
    )
    r = (x0**2 + x1**2) ** 0.5
    mask = torch.minimum(
        torch.tensor(1.0),
        torch.maximum(torch.tensor(0.0), 1 - (r - in_rad) / (out_rad - in_rad)),
    )
    return mask
