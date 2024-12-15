import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import json
import data
import lightning.pytorch as pl
from torch.utils.data import DataLoader
from data.dataset import MyDataset

@data.register("dm_singapo")
class SingapoDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams.update(hparams)

    def _prepare_split(self):
        with open(self.hparams.split_file , "r") as f:
            splits = json.load(f)

        train_ids = splits["train"]
        val_ids = [i for i in train_ids if "augmented" not in i]  
        return train_ids, val_ids

    def _prepare_test_ids(self):
        if self.hparams.get('test_which') == 'acd':
            with open("data/data_acd.json", "r") as f:
                file = json.load(f)
        elif self.hparams.get('test_which') == 'pm':
            with open(self.hparams.split_file, "r") as f:
                file = json.load(f)
        else:
            raise NotImplementedError(f"Dataset {self.hparams.get('test_which')} not implemented for SingapoDataModule")
        ids = file['test']
        return ids
    
    def setup(self, stage=None):
        
        if stage == "fit" or stage is None:
            train_ids, val_ids = self._prepare_split()
            self.train_dataset = MyDataset(self.hparams, model_ids=train_ids, mode="train")
            self.val_dataset = MyDataset(self.hparams, model_ids=val_ids, mode="val")
        elif stage == "validate":
            _, val_ids = self._prepare_split()
            self.val_dataset = MyDataset(self.hparams, model_ids=val_ids, mode="val")
        elif stage == "test":
            test_ids = self._prepare_test_ids()
            self.test_dataset = MyDataset(self.hparams, model_ids=test_ids, mode="test")
        else:
            raise NotImplementedError(f"Stage {stage} not implemented for SingapoDataModule")

    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=128,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            shuffle=False,
            persistent_workers=True
        )
