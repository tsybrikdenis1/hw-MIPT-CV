from pathlib import Path
from typing import Optional

from torch.utils.data import (Dataset, DataLoader)
from torchvision.datasets import ImageFolder
from torchvision.transforms import (
    Compose, Pad, RandomCrop, RandomHorizontalFlip, ToTensor,
)

from pytorch_lightning import LightningDataModule

import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")


class MarketDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        size = 128, 64

        self.transforms_train = Compose([
            Pad(15),
            RandomCrop(size),
            RandomHorizontalFlip(),
            ToTensor(),
        ])

        self.transforms_val = Compose([ToTensor()])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if self.data_train or self.data_val:
            return

        data_dir = Path(self.hparams.data_dir)
        self.data_train = ImageFolder(
            (data_dir / 'train').as_posix(),
            transform=self.transforms_train,
        )
        self.data_val = ImageFolder(
            (data_dir / 'val').as_posix(),
            transform=self.transforms_val,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "market.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
