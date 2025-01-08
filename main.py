import dataclasses
import os
from typing import Literal

import pytorch_lightning as pl
import rich
import tyro
from pytorch_lightning.callbacks import (ModelCheckpoint, RichProgressBar, TQDMProgressBar)
from pytorch_lightning.strategies import SingleDeviceStrategy
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from code.dataset import EMPIARDataset
from code.model import CryoNeRF


@dataclasses.dataclass
class Args:
    """Arguments of CryoNeRF."""
    
    size: int = 256
    """Size of the volume and particle images."""

    batch_size: int = 1
    """Batch size for training."""
    
    ray_num: int = 8192
    """Number of rays to query in a batch."""
    
    enc_dim: int = 16
    """Positional encoding, the output dim is 6 * enc_dim."""
    
    nerf_hid_dim: int = 128
    """Hidden dim of NeRF."""
    
    nerf_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer."""
    
    dfom_hid_dim: int = 128
    """Hidden dim of deformation field."""
    
    dfom_hid_layer_num: int = 2
    """Number of hidden layers besides the input and output layer of deformation field.."""
    
    dfom_encoder_type: Literal["resnet18", "resnet34", "resnet50", "convnext_small", "convnext_base", ""] = "resnet18"
    """Encoder for deformation latent variable."""
    
    dfom_latent_dim: int = 16
    """Latent variable dim for deformation encoder."""
    
    save_dir: str = "experiments/test"
    """Dir to save visualization and checkpoint."""
    
    log_vis_step: int = 1000
    """Number of steps to log visualization."""

    log_density_step: int = 10000
    """Number of steps to log density map."""
    
    print_step: int = 100
    """Number of steps to print once."""
    
    dataset: Literal["empiar-10028", "empiar-10076", "empiar-10049", "empiar-10180",
                     "IgG-1D", "Ribosembly", "Tomotwin-100", "empiar-10345", "empiar-10425",
                     "empiar-10697", "TRPV1", "empiar-11043"] = "empiar-10028"
    """EMPIAR dataset to use."""
    
    dataset_dir: str
    """Root dir for datasets."""
    
    sign: Literal[1, -1] = 1
    """Sign of the particle images."""
    
    seed: int = -1
    """Seed everything"""
    
    load_ckpt: str | None = None
    """The checkpoint to load"""
    
    epochs: int = 1
    """Number of epochs for training."""
    
    enable_dfom: bool = False
    """Whether to enable deformation for heterogeneous reconstruction."""
    
    checkpointing: bool = False
    """Whether to use checkpointing to save GPU memory."""
    
    val_only: bool = False
    """Only val"""
    
    test_only: bool = False
    """Only test"""
    
    first_half: bool = False
    "Whether to use the first half of the data to train."
    
    second_half: bool = False
    "Whether to use the second half of the data to train."
    
    pe_type: Literal["nerf", "cryodrgn", "ingp", "gaussian"] = "cryodrgn"
    "Type of positional encoding to use."
    
    vae_weight: float = 5e-4
    
    precision: str = "16-mixed"
    
    use_vae: bool = False

    cryodrgn_z: str = ""

    dfom_start: int = -1

    dfom_embedding: bool = False

    decode_2d: bool = False

    use_emb: bool = False

    use_vq: bool = False

    use_mixing: bool = False

    volume_mixing: bool = False

    image_mixing: bool = False

    image_only: bool = False

    max_steps: int = -1

    gradient_clip_val: float = None

    # Simple model, which only has the vision encoder and the neural representation
    simple: bool = False

    log_time: bool = False

    hartley: bool = False
    
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
    
    CryoNeRFModel = CryoNeRF
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.dataset == "empiar-10028":
        sign = -1
    elif args.dataset == "empiar-10076":
        sign = 1
    elif args.dataset == "empiar-10049":
        sign = -1
    elif args.dataset == "empiar-10180":
        sign = -1
    elif args.dataset == "IgG-1D":
        sign = -1
    elif args.dataset == "Ribosembly":
        sign = -1
    elif args.dataset == "Tomotwin-100":
        sign = -1
    elif args.dataset == "empiar-10345":
        sign = -1
    elif args.dataset == "empiar-10425":
        sign = -1
    elif args.dataset == "empiar-10697":
        sign = -1
    elif args.dataset == "TRPV1":
        sign = -1
    elif args.dataset == "empiar-11043":
        sign = -1
    else:
        sign = None
        rich.print("[red]Unknown dataset. Use sign specified in args![/red]")
    
    if args.load_ckpt:
        cryo_nerf = CryoNeRFModel.load_from_checkpoint(args.load_ckpt, strict=True, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        cryo_nerf = CryoNeRFModel(args=args)
        
    dataset = EMPIARDataset(
        mrcs=f"{args.dataset_dir}/{args.dataset}/particles.mrcs",
        ctf=f"{args.dataset_dir}/{args.dataset}/ctf.pkl",
        poses=f"{args.dataset_dir}/{args.dataset}/poses.pkl",
        args=args,
        size=args.size, sign=sign if sign is not None else args.sign,
    )

    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True, pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(dataset, batch_size=128, num_workers=16, shuffle=False, pin_memory=True)
        
    logger = WandbLogger(name=f"CryoNeRF-{args.save_dir}", save_dir=args.save_dir, offline=True, project="CryoNeRF")
    logger.experiment.log_code(".")
    
    checkpoint_callback_step = ModelCheckpoint(dirpath=args.save_dir, save_top_k=-1, verbose=True, every_n_train_steps=20000, save_last=True)
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
        print(cryo_nerf)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader, ckpt_path=args.load_ckpt)
    elif args.test_only:
        pass
    else:
        print(cryo_nerf)
        trainer.fit(model=cryo_nerf, train_dataloaders=train_dataloader, ckpt_path=args.load_ckpt)
        validator.validate(model=cryo_nerf, dataloaders=valid_dataloader)