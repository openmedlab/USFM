# Fintune beit for Seg
from typing import Any, List

import torch
import torch.nn as nn
import torchvision.transforms.functional as tvF
from lightning import LightningModule
from mmseg.registry import MODELS
from omegaconf import OmegaConf
from torchmetrics import Dice, MaxMetric, MeanMetric
from torchvision.utils import draw_segmentation_masks

from usfm.data.transforms import DeNormalize
from usfm.models.components.backbone import beit
from usfm.models.components.optim_factory import create_optimizer


class BeitSegLit(LightningModule):
    def __init__(
        self,
        net,
        optimizer: None,
        scheduler: torch.optim.lr_scheduler,
        metric_keys: List[str],
        **args,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.net = MODELS.build(dict(OmegaConf.to_container(net)))
        self.criterion = torch.nn.CrossEntropyLoss()
        self.metric_keys = metric_keys
        self.metric_train = None
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_acc_best = MaxMetric()

        self.train_dice = Dice(num_classes=2, ignore_index=0)
        self.val_dice = Dice(num_classes=2, ignore_index=0)

        self.loss = []
        self.outputs = []
        self.labels = []

    def forward(self, samples):
        return self.net(samples)

    def step(self, batch: Any):
        samples, labels = batch["image"], batch["mask"]
        labels = labels.long()
        extra_features = self.net.backbone(samples)
        decoded = self.net.decode_head(extra_features)
        if self.net.auxiliary_head is not None:
            auxiliary_logits = self.net.auxiliary_head(extra_features)

        # Upsample the output logits to the input resolution
        outputs = nn.functional.interpolate(
            decoded, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        auxiliary_outputs = nn.functional.interpolate(
            auxiliary_logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )

        loss = 1 * self.criterion(outputs, labels) + 0.4 * self.criterion(
            auxiliary_outputs, labels
        )
        return loss, outputs, labels

    def training_step(self, batch: Any, batch_idx: int):
        loss, outputs, labels = self.step(batch)
        self.train_loss(loss)
        self.train_dice(outputs.argmax(dim=1), labels)
        self.log(
            "train/loss",
            self.train_loss,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )
        self.log(
            "train/Dice",
            self.train_dice,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch: Any, batch_idx: int):
        loss, outputs, labels = self.step(batch)
        self.val_loss(loss)
        self.val_dice(outputs.argmax(dim=1), labels)
        self.log(
            "val/loss", self.val_loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )
        self.log(
            "val/Dice", self.val_dice, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
        )
        return loss

    def on_train_epoch_start(self):
        pass

    def on_validation_epoch_end(self) -> None:
        self.make_log(mode="val", epoch_interval=1)

    def make_log(self, mode="train", epoch_interval=20):
        # logger loss
        with torch.no_grad():
            if self.current_epoch % epoch_interval == 0:
                for i, batch in enumerate(self.trainer.datamodule.vis_dataloader()):
                    samples, labels = batch["image"], batch["mask"]
                    samples = samples.to(self.device)
                    labels = labels.to(self.device).long()
                    extra_features = self.net.backbone(samples)
                    decoded = self.net.decode_head(extra_features)
                    if self.net.auxiliary_head is not None:
                        auxiliary_logits = self.net.auxiliary_head(extra_features)

                    # Upsample the output logits to the input resolution
                    outputs = nn.functional.interpolate(
                        decoded, size=labels.shape[-2:], mode="bilinear", align_corners=False
                    )
                    auxiliary_outputs = nn.functional.interpolate(
                        auxiliary_logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )

                    desamples = tvF.convert_image_dtype(DeNormalize(samples.cpu()), torch.uint8)
                    deoutputs = outputs.argmax(1).cpu().to(torch.bool)
                    labels = labels.cpu().to(torch.bool)
                    draw_with_masks = [
                        draw_segmentation_masks(desamples[i], labels[i], alpha=0.1, colors=["red"])
                        for i in range(len(desamples))
                    ]
                    dwm = torch.stack(draw_with_masks, 0)
                    self.loggers[0].experiment.add_images(
                        "labels",
                        dwm.float() / 255,
                        global_step=self.current_epoch,
                        dataformats="NCHW",
                    )

                    draw_with_masks = [
                        draw_segmentation_masks(
                            desamples[i], deoutputs[i], alpha=0.1, colors=["green"]
                        )
                        for i in range(len(desamples))
                    ]
                    dwm = torch.stack(draw_with_masks, 0)
                    self.loggers[0].experiment.add_images(
                        "outputs",
                        dwm.float() / 255,
                        global_step=self.current_epoch,
                        dataformats="NCHW",
                    )
                    break
            else:
                pass

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = create_optimizer(self.hparams.optimizer, self.net)
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "beitSeg.yaml")
    _ = hydra.utils.instantiate(cfg)
