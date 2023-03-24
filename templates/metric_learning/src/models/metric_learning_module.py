from typing import Any

from torch import Tensor, argmax
from torch.nn import Linear
from torch.optim import SGD
from torchvision.models import (resnet18, ResNet18_Weights)

from pytorch_metric_learning.losses import ArcFaceLoss
from pytorch_metric_learning.utils.common_functions import Identity


from .components import LoggedModule


class MetricLearningModule(LoggedModule):
    def __init__(self, num_classes=751, embedding_size=256, lr=0.05, wd=5e-4):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        bacbone_output_size = self.backbone.fc.in_features
        self.backbone.fc = Identity()

        self.embedder = Linear(bacbone_output_size, self.hparams.embedding_size)
        self.arcface = ArcFaceLoss(
            self.hparams.num_classes, self.hparams.embedding_size)

    def forward(self, x: Tensor):
        return self.embedder(self.backbone(x))

    def step(self, batch: Any):
        x, targets = batch
        features = self.forward(x)
        logits = self.arcface.get_logits(features)
        preds = argmax(logits, dim=1)
        loss = self.arcface(features, targets)
        return loss, preds, targets

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log_train(loss, preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        self.log_val(loss, preds, targets)
        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        pass

    def configure_optimizers(self):
        return SGD(
            self.parameters(), self.hparams.lr, momentum=0.9,
            weight_decay=self.hparams.wd, nesterov=True)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "metric_learning.yaml")
    _ = hydra.utils.instantiate(cfg)
