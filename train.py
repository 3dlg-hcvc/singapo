import os
import data 
import systems
import argparse
from utils.misc import load_config
from utils.callbacks import ConfigSnapshotCallback, GPUCacheCleanCallback

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


def train(config):
    pl.seed_everything(42)
    # create datamodule
    dm = data.make(config.data.name, config=config.data)
    # create system
    system = systems.make(
        config.system.name,
        config=config.system,
        load_from_checkpoint=(
            config.cmd_args.ckpt
        ),
    )
    # load pretriained weights of CAGE in our partial model
    system.load_cage_weights(config.cmd_args.pretrained_cage)

    # configure logger
    logger = WandbLogger(**config.logger)

    # configure callbacks
    callbacks = [
        ModelCheckpoint(**config.checkpoint),
        LearningRateMonitor(),
        ConfigSnapshotCallback(config),
        GPUCacheCleanCallback(),
    ]

    # create trainer
    trainer = pl.Trainer(
        devices="auto",
        strategy='ddp',
        accelerator="auto",
        logger=logger,
        callbacks=callbacks,
        **config.trainer
    )

    # start training
    trainer.fit(system, datamodule=dm, ckpt_path=args.ckpt)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml") 
    parser.add_argument(
        "--pretrained_cage", 
        default='pretrained/cage_cfg.ckpt', 
        help="path to the checkpoint of the pretrained CAGE"
    )
    parser.add_argument("--ckpt", default=None, help="path to the weights to be resumed")
    args, extras = parser.parse_known_args()

    assert os.path.exists(args.pretrained_cage), "The pretrained CAGE checkpoint does not exist"
    
    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    # create exp_dir
    os.makedirs(config.logger.save_dir, exist_ok=True)
    os.makedirs(config.checkpoint.dirpath, exist_ok=True)


    train(config)
