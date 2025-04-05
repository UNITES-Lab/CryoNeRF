import os
import random
import time

import matplotlib.pyplot as plt
import mrcfile
import numpy as np
import pytorch_lightning as pl
import rich
import seaborn as sns
import tinycudann as tcnn
import torch
import torch.distributed
import torch.nn as nn
import torch.nn.functional as F
import umap
from einops import rearrange, repeat
from scipy.cluster.vq import kmeans2

from ..utils import *
from .deformation import ImageEncoder


class HashGridNeRF(nn.Module):
    def __init__(self, nerf_hid_dim, nerf_hid_layer_num, hetero_latent_dim=0, dtype=None) -> None:
        super().__init__()

        self.hash_grid_config = {
            "otype": "HashGrid",
            "n_levels": 16,
            "n_features_per_level": 2,
            "log2_hashmap_size": 19,
            "base_resolution": 16,
            "per_level_scale": 1.4472692012786865,
        }
        self.mlp_config = {
            "otype": "FullyFusedMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": nerf_hid_dim,
            "n_hidden_layers": nerf_hid_layer_num,
        } if nerf_hid_dim <= 128 else {
            "otype": "CutlassMLP",
            "activation": "ReLU",
            "output_activation": "None",
            "n_neurons": nerf_hid_dim,
            "n_hidden_layers": nerf_hid_layer_num,
        }

        self.dtype = dtype
        self.hash_grid_encoding = tcnn.Encoding(3, self.hash_grid_config, dtype=dtype, seed=random.randint(0, 1524367))
        self.mlp = tcnn.Network(self.hash_grid_encoding.n_output_dims + hetero_latent_dim, 1, self.mlp_config, seed=random.randint(0, 1524367))

    def forward(self, coords, latent_variable=None):
        orig_shape = coords.shape

        encoded_pos = self.hash_grid_encoding(coords.reshape(-1, 3))

        if latent_variable is not None:
            latent_variable = rearrange(repeat(
                latent_variable, "N C -> N H W C", H=orig_shape[1], W=orig_shape[2]), "N H W C -> (N H W) C")
            density = self.mlp(torch.cat([encoded_pos, latent_variable], dim=1)).reshape(
                *orig_shape[:-1], -1)
        else:
            density = self.mlp(encoded_pos).reshape(*orig_shape[:-1], -1)

        density = density.float()

        return density


class CryoNeRF(pl.LightningModule):
    def __init__(self, args) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.args = args
        self.size = args.size
        self.batch_size = args.batch_size
        self.ray_num = args.ray_num
        self.nerf_hid_dim = args.nerf_hid_dim
        self.nerf_hid_layer_num = args.nerf_hid_layer_num
        self.hetero_encoder_type = args.hetero_encoder_type
        self.hetero_latent_dim = args.hetero_latent_dim
        self.save_dir = args.save_dir
        self.log_vis_step = args.log_vis_step
        self.log_density_step = args.log_density_step
        self.print_step = args.print_step
        self.hetero = args.hetero

        if self.hetero:
            self.deformation_encoder = ImageEncoder(self.hetero_encoder_type, self.hetero_latent_dim if self.hetero else 0,
                                                    size=self.size, hartley=self.args.hartley)

        self.nerf = HashGridNeRF(self.nerf_hid_dim, self.nerf_hid_layer_num, self.hetero_latent_dim if self.hetero else 0,
                                 dtype=torch.float if "32" in self.args.precision else None)

    def configure_optimizers(self):
        param_group = [{"params": self.nerf.parameters(), "lr": 1e-4}]
        if self.hetero:
            param_group.append(
                {"params": self.deformation_encoder.parameters(), "lr": 1e-4})

        optimizer = torch.optim.AdamW(param_group)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_steps, eta_min=1e-5)

        return {"optimizer": optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}}

    def render_density(self, coords_query, latent_variable=None):
        pred_density = self.nerf(coords_query, latent_variable)

        return pred_density

    def on_train_start(self) -> None:
        if self.args.log_time:
            self.time_start = time.time()
            
        self.ray_idx_all = repeat(torch.arange(
            self.size**2), "HW -> B HW D Dim3", B=self.batch_size, D=self.size, Dim3=3).long().cuda()
            
        x = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        y = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        z = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        self.volume_grid = torch.from_numpy(np.stack([coord.flatten()
                                            for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).cuda()
        if "32" in self.args.precision:
            self.volume_grid = self.volume_grid.float()
        elif "16" in self.args.precision:
            self.volume_grid = self.volume_grid.half()

        self.t_scale = x[-1] - x[0]
            
        self.raw_size = self.trainer.train_dataloader.dataset.raw_size
        self.Apix = self.trainer.train_dataloader.dataset.Apix

    def training_step(self, batch, batch_idx):
        R = batch["rotations"]
        t = batch["translations"]
        N = R.shape[0]

        volume_grid_query = repeat(self.volume_grid, "HWD Dim3 -> B HWD Dim3", B=R.shape[0]).bmm(R) + (self.t_scale * t.unsqueeze(1)).bmm(R)
        volume_grid_query = volume_grid_query.reshape(N, self.size**2, self.size, 3)

        pred_density = []

        if self.hetero:
            latent_variable = self.deformation_encoder(batch["enc_images"].unsqueeze(1))
        else:
            latent_variable = None

        for ray_idx in torch.split(self.ray_idx_all, self.ray_num, dim=1):
            sampled_coords_xyz = torch.gather(volume_grid_query, 1, ray_idx)
            pred_density_block = self.render_density(sampled_coords_xyz, latent_variable)
            pred_density.append(pred_density_block.squeeze(-1))

        pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
        pred_image = pred_density.mean(-1)
        corrupted_pred_image = torch.fft.fftshift(
            torch.fft.irfft2(
                torch.fft.rfft2(torch.fft.ifftshift(pred_image)) * torch.fft.fftshift(batch["ctfs"])[..., :self.size // 2 + 1]
            )
        )

        loss_recon = F.mse_loss(corrupted_pred_image, batch["images"])
        loss = loss_recon

        if self.global_step % self.print_step == 0 and self.trainer.global_rank == 0:
            result = f"Current step: {self.global_step:06d}, " + \
                f"loss: {loss.item():.6f}, " + \
                f"loss_recon: {loss_recon.item():.6f}"
            if latent_variable is not None:
                result += f", latent_norm: {torch.norm(latent_variable, p=2, dim=-1).mean().item():.6f}"
            rich.print(result)

        if self.global_step % self.log_vis_step == 0 and self.trainer.global_rank == 0:
            log_dir = f"{self.save_dir}/vis/{self.global_step:06d}"
            os.makedirs(log_dir, exist_ok=True)
            plt.imsave(f"{log_dir}/{self.global_step:06d}_pr.png", pred_image[0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/{self.global_step:06d}_cr.png", corrupted_pred_image[0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/{self.global_step:06d}_gt.png", batch["images"][0].numpy(force=True), cmap="gray")
            plt.imsave(f"{log_dir}/{self.global_step:06d}_x.png",
                       pred_density[0, self.size // 2].numpy(force=True).transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/{self.global_step:06d}_y.png",
                       pred_density[0, :, self.size // 2].numpy(force=True).transpose(), cmap="gray")
            plt.imsave(f"{log_dir}/{self.global_step:06d}_z.png",
                       pred_density[0, :, :, self.size // 2].numpy(force=True).transpose(), cmap="gray")

        if self.global_step % self.log_density_step == 0 and self.trainer.global_rank == 0:
            log_dir = f"{self.save_dir}/vis/{self.global_step:06d}"
            os.makedirs(log_dir, exist_ok=True)

            if not self.args.log_time:
                file_name = f"{log_dir}/{self.global_step:06d}_volume.mrc"

                with mrcfile.new(file_name, overwrite=True) as mrc:
                    density = pred_density[0].numpy(force=True).astype(np.float32)
                    density = np.rot90(density, k=3, axes=(1, 2))
                    density = np.rot90(density, k=2, axes=(0, 2))
                    density = np.rot90(density, k=3, axes=(0, 1))
                    mrc.set_data(density)
                    mrc.set_volume()
            else:
                time_elapsed = time.time() - self.time_start
                file_name = f"{log_dir}/{self.global_step:06d}_{time_elapsed:.4f}_volume.mrc"

                ray_idx_all = repeat(torch.arange(self.size**2), "HW -> B HW D Dim3", B=1, D=self.size, Dim3=3).long().cuda()
                volume_grid_query = self.volume_grid.unsqueeze(0).reshape(1, self.size**2, self.size, 3)

                pred_density = []
                for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
                    sampled_coords_xyz = torch.gather(volume_grid_query, 1, ray_idx)
                    pred_density_block = self.render_density(sampled_coords_xyz).detach().cpu()
                    pred_density.append(pred_density_block.squeeze(-1))

                pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
                with mrcfile.new(file_name, overwrite=True) as mrc:
                    density = pred_density[0].numpy(force=True).astype(np.float32)
                    density = np.rot90(density, k=3, axes=(1, 2))
                    density = np.rot90(density, k=2, axes=(0, 2))
                    density = np.rot90(density, k=3, axes=(0, 1))
                    mrc.set_data(density)
                    mrc.set_volume()
                    mrc.voxel_size = self.Apix
                    mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0

        return loss

    def on_validation_epoch_start(self):
        self.latent_vectors = []
        self.umap_model = umap.UMAP(n_components=2 if self.args.embedding == "2d" else 1)
        x = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        y = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        z = np.linspace(-0.5, 0.5, self.size, endpoint=False)
        self.volume_grid_query = torch.from_numpy(
            np.stack([coord.flatten() for coord in np.meshgrid(x, y, z, indexing='xy')], axis=1)).float().cuda().unsqueeze(0)
        self.volume_grid_query = self.volume_grid_query.reshape(1, self.size**2, self.size, 3)
        self.raw_size = self.trainer.val_dataloaders.dataset.raw_size
        self.Apix = self.trainer.val_dataloaders.dataset.Apix

    @torch.no_grad
    def validation_step(self, batch, batch_idx):
        if self.hetero:
            self.latent_vectors.append(self.deformation_encoder(batch["enc_images"].unsqueeze(1)))
        else:
            return

    @torch.no_grad
    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            if self.hetero:
                latent_variables = torch.cat(self.latent_vectors, dim=0)

                if latent_variables.shape[1] < 2:
                    data_1d = latent_variables.numpy(force=True).flatten()
                    
                    plt.figure(figsize=(6, 4))
                    sns.histplot(data_1d, bins=250, color='blue', alpha=0.7)
                    plt.xlabel("Value")
                    plt.ylabel("Count")
                    plt.title("1D Latent Variable Distribution")
                    plt.tight_layout()
                    plt.savefig(f'{self.save_dir}/latent.png', dpi=300)
                    plt.close()
        
                    np.save(f"{self.save_dir}/latent_variables.npy", latent_variables.numpy(force=True))
                    return
                elif self.args.embedding == "2d":
                    latent_2d = self.umap_model.fit_transform(latent_variables.numpy(force=True))
                    np.save(f"{self.save_dir}/latent_2d.npy", latent_2d)
                    centroids, labels = kmeans2(latent_2d, k=6, seed=42)
                    latent_2d_with_labels = np.column_stack((latent_2d, labels))

                    sns.scatterplot(x=latent_2d_with_labels[:, 0], y=latent_2d_with_labels[:, 1],
                                    hue=latent_2d_with_labels[:, 2], palette="viridis")
                    plt.xlabel("UMAP1")
                    plt.ylabel("UMAP2")
                    plt.tight_layout()
                    plt.savefig(f'{self.save_dir}/scatter_plot.png', dpi=300)
                    plt.close()

                    fig = sns.jointplot(x=latent_2d[:, 0], y=latent_2d[:, 1], kind="hex")
                    fig.ax_joint.set_xlabel("UMAP1")
                    fig.ax_joint.set_ylabel("UMAP2")
                    plt.tight_layout()
                    plt.savefig(f'{self.save_dir}/latent.png', dpi=300)
                    plt.close()
                    np.save(f"{self.save_dir}/latent_variables.npy", latent_variables.numpy(force=True))
            else:
                latent_variables = None

            ray_idx_all = repeat(torch.arange(self.size**2), "HW -> B HW D Dim3", B=1, D=self.size, Dim3=3).long().cuda()

            if self.hetero:
                for i, label in enumerate(np.unique(labels)):
                    latent_variables_for_label = latent_variables[torch.from_numpy(labels == label)].mean(dim=0)

                    pred_density = []
                    for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
                        sampled_coords_xyz = torch.gather(self.volume_grid_query, 1, ray_idx)
                        pred_density_block = self.render_density(sampled_coords_xyz,
                                                                 latent_variables_for_label.unsqueeze(0).cuda()).detach().cpu()
                        pred_density.append(pred_density_block.squeeze(-1))
                    pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)

                    with mrcfile.new(f"{self.save_dir}/volume_{i}.mrc", overwrite=True) as mrc:
                        density = pred_density[0].numpy(force=True).astype(np.float32)
                        density = np.rot90(density, k=3, axes=(1, 2))
                        density = np.rot90(density, k=2, axes=(0, 2))
                        density = np.rot90(density, k=3, axes=(0, 1))
                        mrc.set_data(density[::-1, :, :])
                        mrc.set_volume()
                        mrc.voxel_size = self.Apix
                        mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0
            else:
                pred_density = []
                for ray_idx in torch.split(ray_idx_all, self.ray_num, dim=1):
                    sampled_coords_xyz = torch.gather(self.volume_grid_query, 1, ray_idx)
                    pred_density_block = self.render_density(sampled_coords_xyz).detach().cpu()
                    pred_density.append(pred_density_block.squeeze(-1))

                pred_density = rearrange(torch.cat(pred_density, dim=1), "B (H W) D -> B H W D", H=self.size, W=self.size)
                with mrcfile.new(f"{self.save_dir}/volume.mrc", overwrite=True) as mrc:
                    density = pred_density[0].numpy(force=True).astype(np.float32)
                    density = np.rot90(density, k=3, axes=(1, 2))
                    density = np.rot90(density, k=2, axes=(0, 2))
                    density = np.rot90(density, k=3, axes=(0, 1))
                    mrc.set_data(density)
                    mrc.set_volume()
                    mrc.voxel_size = self.Apix
                    mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z = 0, 0, 0
