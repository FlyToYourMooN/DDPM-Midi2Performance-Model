import copy
import logging
import os

import hydra
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader
import torch
from models.callbacks import EMAWeightUpdate
from models.diffusion import (DDPM, DDPMWrapper, SuperResModel)
import matplotlib.pyplot as plt
from recon_dataset import ReconstructionDataset
from hydra.utils import get_original_cwd

logger = logging.getLogger(__name__)


def __parse_str(s):
    split = s.split(",")
    return [int(s) for s in split if s != "" and s is not None]


@hydra.main(config_path="configs", config_name="train")
def train(config):
    # Get config and setup

    logger.info(OmegaConf.to_yaml(config))

    # Set seed
    seed_everything(config.training.seed, workers=True)

    # Dataset
    train_dataset = ReconstructionDataset(config.data.train_root, config.data)
    #plt.imshow(train_dataset[0][1][0])
    #plt.savefig("2.png")
    #exit()
    N = len(train_dataset)
    batch_size = config.training.batch_size
    batch_size = min(N, batch_size)

    val_dataset = ReconstructionDataset(config.data.val_root, config.data)

    # Model
    lr = config.training.lr
    dim_mults = __parse_str(config.model.dim_mults)

    # Use the superres model for conditional training
    decoder_cls = SuperResModel
    decoder = decoder_cls(
        in_channels=config.data.n_channels,
        model_channels=config.model.dim,
        out_channels=1,
        num_res_blocks=config.model.n_residual,
        channel_mult=dim_mults,
        use_checkpoint=False,
        dropout=config.model.dropout,
        dims = 2
    )
    #decoder = torch.compile(decoder, mode="max-autotune") I think its fast enough
    # EMA parameters are non-trainable
    ema_decoder = copy.deepcopy(decoder)
    for p in ema_decoder.parameters():
        p.requires_grad = False

    online_ddpm = DDPM(
        decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )
    target_ddpm = DDPM(
        ema_decoder,
        beta_1=config.model.beta1,
        beta_2=config.model.beta2,
        T=config.model.n_timesteps,
    )

    ddpm_wrapper = DDPMWrapper(
        online_ddpm,
        target_ddpm,
        lr=lr,
        n_anneal_steps=config.training.n_anneal_steps,
        loss=config.training.loss,
        conditional=True,
        grad_clip_val=config.training.grad_clip,
    )

    # Trainer
    train_kwargs = {}


    original_cwd = get_original_cwd()
    results_dir = os.path.join(original_cwd, config.training.results_dir)

    chkpt_callback = ModelCheckpoint(
        dirpath=os.path.join(results_dir, "checkpoints"),
        filename=f"ddpmv2-{config.training.chkpt_prefix}" + "-{epoch:02d}-{loss:.4f}",
        every_n_epochs=config.training.chkpt_interval,
        save_on_train_epoch_end=True,
    )

    train_kwargs["default_root_dir"] = results_dir
    train_kwargs["max_epochs"] = config.training.epochs
    train_kwargs["log_every_n_steps"] = config.training.log_step
    train_kwargs["callbacks"] = [chkpt_callback]

    if config.training.use_ema:
        ema_callback = EMAWeightUpdate(tau=config.training.ema_decay)
        train_kwargs["callbacks"].append(ema_callback)

    loader_kws = {}
    train_kwargs["devices"] = [0]
    train_kwargs["accelerator"] = "gpu"
    
    # Half precision training
    if config.training.fp16:
        train_kwargs["precision"] = 16

    # Loader
    train_loader = DataLoader(
        train_dataset,
        batch_size,
        num_workers=config.training.workers,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        **loader_kws,
    )
    val_loader = DataLoader(
        val_dataset,
        1,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        **loader_kws,
    )
    
    train_kwargs["limit_val_batches"] = 1
    
    trainer = pl.Trainer(**train_kwargs)
    if config.training.ckpt_path:
        trainer.fit(ddpm_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=config.training.ckpt_path)
    else:
        trainer.fit(ddpm_wrapper, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    train()
