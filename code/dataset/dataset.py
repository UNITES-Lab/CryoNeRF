import os
import pickle
import random

# import cv2
import mrcfile
import numpy as np
import rich
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from skimage.draw import disk
from skimage.filters import butterworth
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset

from ..utils import *


def fft_resize(image, raw_size, new_size):
    start = int(raw_size / 2 - new_size / 2)
    stop = int(raw_size / 2 + new_size / 2)
    
    oldft = ht2_center(image)
    newft = oldft[start:stop, start:stop]
    new = iht2_center(newft)
    
    return new


class EMPIARDataset(Dataset):
    def __init__(self, mrcs: str, ctf: str, poses: str, args, size=256, sign=1) -> None:
        super().__init__()
        self.size = size
        self.args = args
        self.sign = sign

        with open(poses, "rb") as f:
            poses = pickle.load(f)
        self.rotations, self.translations = poses

        with open(ctf, "rb") as f:
            self.ctf_params = pickle.load(f)

        with mrcfile.open(mrcs) as f:
            # mean = np.mean(f.data)
            # std = np.std(f.data)
            # print(f"Normalizing images using mean: {mean}, std: {std}")
            # self.images = sign * (f.data - mean) / std
            self.images = f.data
            
        # local_rng = np.random.default_rng(42)
        # permuted_indices = local_rng.permutation(np.arange(len(self.images)))
        # self.images = self.images[permuted_indices][:200000]
        # self.ctf_params = self.ctf_params[permuted_indices][:200000]
        # self.rotations = self.rotations[permuted_indices][:200000]
        # self.translations = self.translations[permuted_indices][:200000]
            
        # first randomly permute and then split
        if args.first_half or args.second_half:
            local_rng = np.random.default_rng(42)
            permuted_indices = local_rng.permutation(np.arange(len(self.images)))
            self.images = self.images[permuted_indices]
            self.ctf_params = self.ctf_params[permuted_indices]
            self.rotations = self.rotations[permuted_indices]
            self.translations = self.translations[permuted_indices]

        if args.first_half:
            self.images = self.images[:len(self.images) // 2]
            self.ctf_params = self.ctf_params[:len(self.ctf_params) // 2]
            self.rotations = self.rotations[:len(self.rotations) // 2]
            self.translations = self.translations[:len(self.translations) // 2]
        elif args.second_half:
            self.images = self.images[len(self.images) // 2:]
            self.ctf_params = self.ctf_params[len(self.ctf_params) // 2:]
            self.rotations = self.rotations[len(self.rotations) // 2:]
            self.translations = self.translations[len(self.translations) // 2:]
            
        self.raw_size = self.ctf_params[0, 0]
        self.Apix = self.ctf_params[0, 1] * self.ctf_params[0, 0] / self.size
        self.img_mask = window_mask(self.size, in_rad=0.8, out_rad=0.95)

        if args.cryodrgn_z:
            with open(args.cryodrgn_z, "rb") as f:
                self.latent_variables = pickle.load(f)

    def __len__(self):
        if self.args.max_steps == -1:
            return len(self.images)
        else:
            return self.args.max_steps * self.args.batch_size

    def __getitem__(self, index) -> dict:
        if self.args.max_steps > len(self.images):
            index = random.randint(0, len(self.images) - 1)
        sample = {}
        
        sample["rotations"] = torch.from_numpy(self.rotations[index]).float()
        sample["translations"] = torch.from_numpy(np.concatenate([self.translations[index], np.array([0])])).float()
        sample["images"] = torch.from_numpy(resize(self.images[index].copy(), (self.size, self.size), order=1)).float() * self.sign
        
        if self.args.dataset == "IgG-1D" or self.args.dataset == "Ribosembly" or self.args.dataset == "Tomotwin-100":
            sample["images"] /= 255
            
        sample["ctf_params"] = torch.from_numpy(self.ctf_params[index]).float()

        freq_v = np.fft.fftshift(np.fft.fftfreq(self.size))
        freq_h = np.fft.fftshift(np.fft.fftfreq(self.size))
        freqs = torch.from_numpy(np.stack([freq.flatten() for freq in np.meshgrid(freq_v, freq_h, indexing="ij")],
                                          axis=1)) / (sample["ctf_params"][1] * sample["ctf_params"][0] / self.size)

        rr, cc = disk((self.size // 2, self.size // 2), self.size * 0.6 // 2, shape=(self.size, self.size))
        freqs_mask = np.zeros((self.size, self.size))
        freqs_mask[rr, cc] = 1
        sample["freqs_mask"] = torch.from_numpy(freqs_mask).float()
        # sample["ctfs"] = compute_ctf(freqs, *torch.split(sample["ctf_params"][2:], 1, 0)).reshape(sample["images"].shape).float() * sample["freqs_mask"]
        sample["ctfs"] = compute_ctf(freqs, *torch.split(sample["ctf_params"][2:], 1, 0)).reshape(sample["images"].shape).float()
        # sample["enc_images"] = sample["images"]
        # sample["enc_images"] = fft.ht2_center(torch.from_numpy(
        #     cv2.resize(self.images[index].copy(), (self.size, self.size),  interpolation=cv2.INTER_LINEAR))).float() * sample["freqs_mask"] / 320.8006591796875 #* sample["ctfs"].sign() / 320.8006591796875
        # sample["enc_images"] = torch.from_numpy(low_pass_filter(sample["images"].numpy(), self.size, scale=0.6)).float()
        # sample["enc_images"] = torch.from_numpy(butterworth(sample["images"].numpy(), cutoff_frequency_ratio=6 / self.size,
        #                                                     high_pass=False, order=1)).float()
        if self.args.hartley:
            sample["enc_images"] = symmetrize_ht(self.sign * ht2_center(sample["images"]))
        else:
            sample["enc_images"] = self.sign * sample["images"]
            # sample["enc_images"] = torch.from_numpy(butterworth(sample["images"].numpy(), cutoff_frequency_ratio=6 / self.size,
            #                                                     high_pass=False, order=1)).float()

        sample["img_mask"] = self.img_mask
        # sample["sphere_mask"] = torch.from_numpy(draw_inscribed_sphere(self.size)).reshape(self.size**2, self.size).bool()
        
        if self.args.cryodrgn_z:
            sample["latent_variables"] = self.latent_variables[index]
        sample["indices"] = index
        
        return sample
