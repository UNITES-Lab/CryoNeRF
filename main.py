import dataclasses
import os
from code.dataset import EMPIARDataset
from code.model import CryoNeRF
from typing import Literal

import pytorch_lightning as pl
import rich
import torch
import tyro
from pytorch_lightning.callbacks import (ModelCheckpoint, RichProgressBar,
                                         TQDMProgressBar)
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.utilities import rank_zero
from torch.utils.data import DataLoader


@dataclasses.dataclass
class Args:
    """Arguments of CryoNeRF."""
    
    dataset_dir: str
    """Root dir for datasets. It should be the parent folder of the dataset you want to reconstruct."""
    
    dataset: Literal["empiar-10028", "empiar-10076", "empiar-10049", "empiar-10180", "IgG-1D", "Ribosembly",
                     "uniform", "cooperative", "noncontiguous", ""] = ""
    """Which dataset to use. Default as "" for new datasets."""
    
    size: int = 256
    """Size of the volume and particle images."""

    batch_size: int = 1
    """Batch size for training."""
    
    ray_num: int = 8192
    """Number of rays to query in a batch."""
    
    nerf_hid_dim: int = 128
    """Hidden dim of NeRF."""
    
    nerf_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer."""
    
    hetero_encoder_type: Literal["resnet18", "resnet34", "resnet50", "convnext_small", "convnext_base", ""] = "resnet34"
    """Encoder for deformation latent variable."""
    
    hetero_latent_dim: int = 16
    """Latent variable dim for deformation encoder."""
    
    save_dir: str = "experiments/test"
    """Dir to save visualization and checkpoint."""
    
    log_vis_step: int = 1000
    """Number of steps to log visualization."""

    log_density_step: int = 10000
    """Number of steps to log a density map."""
    
    ckpt_save_step: int = 20000
    """Number of steps to save a checkpoint."""
    
    print_step: int = 100
    """Number of steps to print once."""
    
    sign: Literal[1, -1] = -1
    """Sign of the particle images. For datasets used in the paper, this will be automatically set."""
    
    load_to_mem: bool = False
    """Whether to load the full dataset to memory. This can cost a large amount of memory."""
    
    seed: int = -1
    """Whether to set a random seed. Default to not."""
    
    load_ckpt: str | None = None
    """The checkpoint to load"""
    
    epochs: int = 1
    """Number of epochs for training."""
    
    hetero: bool = False
    """Whether to enable heterogeneous reconstruction."""
    
    val_only: bool = False
    """Only val"""
    
    first_half: bool = False
    """Whether to use the first half of the data to train for GSFSC computation."""
    
    second_half: bool = False
    """Whether to use the second half of the data to train for GSFSC computation."""
    
    precision: str = "16-mixed"
    """The neumerical precision for all the computation. Recommended to set as default at 16-mixed."""

    max_steps: int = -1
    """The number of training steps. If set, this will supersede num_epochs."""

    log_time: bool = False
    """Whether to log the training time."""

    hartley: bool = True
    """Whether to encode the particle image in hartley space. This will improve heterogeneous reconstruction."""
    
    embedding: Literal["2d", "1d"] = "2d"
    """Whether to use scalar embeddings for particle images."""
    
class IterationProgressBar(TQDMProgressBar):
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        if self.trainer.max_steps:
            bar.total = self.trainer.max_steps
        else:
            bar.total = self.trainer.num_training_batches
        return bar

    def on_train_epoch_start(self, trainer, pl_module):
        # Only reset if max_steps is not set
        if not self.trainer.max_steps:
            super().on_train_epoch_start(trainer, pl_module)

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.total = self.trainer.num_val_batches[0] 
        return bar
    
    
class RichIterationProgressBar(RichProgressBar):
    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.is_disabled:
            return
        
        if trainer.max_steps > -1:
            total_batches = trainer.max_steps
        else:
            total_batches = self.total_train_batches
            
        train_description = "Training..."

        if self.train_progress_bar_id is not None and self._leave:
            self._stop_progress()
            self._init_progress(trainer)
        if self.progress is not None:
            if self.train_progress_bar_id is None:
                self.train_progress_bar_id = self._add_task(total_batches, train_description)
            else:
                self.progress.reset(
                    self.train_progress_bar_id,
                    total=total_batches,
                    description=train_description,
                    visible=True,
                )

        self.refresh()
        
    def get_metrics(self, trainer, model):
        # don't show the version number
        items = super().get_metrics(trainer, model)
        items.pop("v_num", None)
        return items
    

if __name__ == "__main__":
    args = tyro.cli(Args)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    sign_map = {
        "empiar-10028": -1,
        "empiar-10076": 1,
        "empiar-10049": -1,
        "empiar-10180": -1,
        "IgG-1D": -1,
        "Ribosembly": -1,
        "uniform": 1,
        "cooperative": 1,
        "noncontiguous": 1
    }

    sign = sign_map.get(args.dataset, None) or args.sign
    
    if args.load_ckpt:
        cryo_nerf = CryoNeRF.load_from_checkpoint(args.load_ckpt, strict=True, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        cryo_nerf = CryoNeRF(args=args)
        
    dataset = EMPIARDataset(
        mrcs=os.path.join(args.dataset_dir, "particles.mrcs"),
        ctf=os.path.join(args.dataset_dir, "ctf.pkl"),
        poses=os.path.join(args.dataset_dir, "poses.pkl"),
        args=args,
        size=args.size,
        sign=sign,
    )
    rich.print("[green]Dataset loaded.")

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=128, num_workers=16, shuffle=False, pin_memory=True)
        
    logger = WandbLogger(name=f"CryoNeRF-{args.save_dir}", save_dir=args.save_dir, offline=True, project="CryoNeRF")
    logger.experiment.log_code(".")
    
    checkpoint_callback_step = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_train_steps=args.ckpt_save_step, save_last=True)
    checkpoint_callback_epoch = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_epochs=1)

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="auto",
        max_epochs=args.epochs if args.max_steps == -1 else None,
        max_steps=args.max_steps,
        logger=logger,
        callbacks=[RichIterationProgressBar(), checkpoint_callback_step, checkpoint_callback_epoch],
        precision=args.precision,
    )

    validator = pl.Trainer(
        accelerator="gpu",
        strategy=SingleDeviceStrategy(device="cuda:0"),
        max_epochs=args.epochs,
        logger=None,
        enable_checkpointing=False,
        enable_model_summary=False,
        devices=1,
        callbacks=[RichIterationProgressBar()],
        precision=args.precision,
    )
    
    if args.val_only:
        if trainer.is_global_zero:
            print(cryo_nerf)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader, ckpt_path=args.load_ckpt)
    else:
        if trainer.is_global_zero:
            print(cryo_nerf)
        trainer.fit(model=cryo_nerf, train_dataloaders=train_dataloader, ckpt_path=args.load_ckpt)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader)