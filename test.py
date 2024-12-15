import os
import data
import torch
import systems
import argparse
import lightning.pytorch as pl
from utils.misc import load_config


def test(config):

    dm = data.make(config.data.name, config=config.data)
    system = systems.make(config.system.name, config=config.system)

    trainer = pl.Trainer(devices='auto',
                        strategy='ddp', 
                        accelerator='auto',
                        logger=False,
                        **config.trainer)
    
    checkpoint = torch.load(config.cmd_args.ckpt)
    trainer.fit_loop.load_state_dict(checkpoint['loops']['fit_loop'])
    trainer.test(system, datamodule=dm, ckpt_path=config.cmd_args.ckpt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to the config file')
    parser.add_argument('--ckpt', required=True, help='path to the weights to do prediction')  
    parser.add_argument('--which_data', type=str, default='pm', choices=['pm', 'acd'], help='which dataset to test on')
    parser.add_argument('--label_free', action="store_true", help='whether to test on object category label-free mode')
    parser.add_argument('--G_dir', type=str, default=None, help='path to the directory containing predicted graphs if testing on predicted graphs')
    parser.add_argument('--no_GT', action="store_true", help='turn on if there is no ground truth object available')


    args, extras = parser.parse_known_args()

    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    config.data.test_which = args.which_data
    config.system.test_which = args.which_data

    if args.no_GT:
        config.data.test_no_GT = True
        config.system.test_no_GT = True

    if args.G_dir is not None:
        assert os.path.exists(args.G_dir), f'Path to the predicted graphs does not exist: {args.G_dir}'
        config.data.test_pred_G = True
        config.data.G_dir = args.G_dir
        config.system.test_pred_G = True
    
    if args.label_free:
        config.data.test_label_free = True
        config.system.test_label_free = True
    
    test(config)