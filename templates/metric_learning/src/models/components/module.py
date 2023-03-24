from typing import (Any, List)

from pytorch_lightning import LightningModule
from torchmetrics import (MaxMetric, MeanMetric, Accuracy)


class LoggedModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.val_acc_best = MaxMetric()

    def on_train_start(self):
        self.val_acc_best.reset()

    def log_train(self, loss, preds, targets):
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

    def log_val(self, loss, preds, targets):
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
