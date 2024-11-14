import datetime
import glob
import os
import re
import shutil
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from timm.utils import AverageMeter
from torchmetrics.segmentation import GeneralizedDiceScore, MeanIoU

from usdsgen.utils.file_manager import Top_K_results_manager
from usdsgen.utils.metrics import get_seg_fromarray
from usdsgen.utils.modelutils import get_grad_norm

from .basetrainer import BaseTrainer


def save_image(mask, path_dir, mask_path):
    file_name = os.path.join(path_dir, os.path.basename(mask_path))
    Image.fromarray(mask).save(file_name)


def save_seg_pre_gt(output_path, result, epoch, max_dice):
    # save the new checkpoint and folder
    mask_path_dir = os.path.join(output_path, f"best{epoch}_dice{max_dice:.3f}")

    mask_pre_path_dir = os.path.join(mask_path_dir, "mask_pre")
    mask_gt_path_dir = os.path.join(mask_path_dir, "mask_gt")
    os.makedirs(mask_path_dir, exist_ok=True)
    os.makedirs(mask_pre_path_dir, exist_ok=True)
    os.makedirs(mask_gt_path_dir, exist_ok=True)

    mask_pre_all = result["mask_pre_all"].numpy().astype(np.uint8)
    mask_path_all = result["mask_path_all"]
    mask_gt_all = result["mask_gt_all"].numpy().astype(np.uint8)
    for mask_gt, mask_pre, mask_path in zip(mask_gt_all, mask_pre_all, mask_path_all):
        save_image(mask_gt, mask_gt_path_dir, mask_path)
        save_image(mask_pre, mask_pre_path_dir, mask_path)

    return mask_path_dir


def save_segmetrics(
    output_path, val_result, epoch, max_dice, mask_path_dir, isbest=False
):
    # save segmetrics
    checkpoint_name = None
    list_dict = {
        k: (v[0].item(), v[1].item()) for k, v in val_result["segmetrics"].items()
    }
    df_metrics = pd.DataFrame.from_dict(list_dict, orient="index").reset_index()
    df_metrics.insert(0, "epoch", epoch)
    df_metrics.columns = ["epoch", "metrics", "mean", "std"]
    df_metrics.to_csv(os.path.join(output_path, "allsegmetrics.csv"), mode="a")
    if isbest:
        df_metrics.to_csv(os.path.join(mask_path_dir, "segmetrics.csv"))
        # find the "best*.pth" and remove it
        for bestpth in glob.glob(os.path.join(output_path, "best*.pth")):
            os.remove(bestpth)
        checkpoint_name = f"best{epoch}.pth"
    return checkpoint_name


class SegTrainer(BaseTrainer):
    def __init__(self, config: DictConfig) -> None:
        # base setting
        super().__init__(config)
        self.max_dice = 0.0
        # task specific setting
        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "max_dice": self.max_dice,
            "scaler": self.scaler,
            "epoch": self.epoch,
            "config": self.config,
        }
        self.top_k_results_manager = Top_K_results_manager(mode="max", max_len=5)
        self.Dice = GeneralizedDiceScore(
            num_classes=self.config.data.num_classes,
            include_background=False,
            weight_type="linear",
            input_format="index",
        ).to(self.fabric.device)
        self.IOU = MeanIoU(
            num_classes=self.config.data.num_classes,
            include_background=False,
            input_format="index",
        ).to(self.fabric.device)

    def fit(self):
        self.logger.info("Start fiting")
        self.check_resume()
        self.load_resume()
        isbest = False
        start_time = time.time()

        start_epoch = max(self.config.train.start_epoch, self.epoch)

        for epoch in range(start_epoch, self.config.train.epochs):
            self.epoch = epoch
            train_dice, train_iou, train_loss = self.train_one_epoch(
                self.dataloader_train
            )

            # train log
            tensorboard_log = {
                "loss": {"train_loss": train_loss},
                "dice": {"train_dice": train_dice},
                "iou": {"train_iou": train_iou},
            }

            # validation and save the best model
            if epoch % self.config.train.val_freq == 0:
                val_dice, val_iou, val_loss, val_result = self.validate(
                    self.dataloader_val
                )
                tensorboard_log["loss"]["val_loss"] = val_loss
                tensorboard_log["dice"]["val_dice"] = val_dice
                tensorboard_log["iou"]["val_iou"] = val_iou

                self.logger.info(
                    f"Dice of the network on all test images: {val_dice:.3f}"
                )
                self.fabric.barrier()
                self.fabric.all_reduce([train_loss, val_loss])
                self.fabric.all_reduce([train_dice, val_dice])
                self.fabric.all_reduce([train_iou, val_iou])
                val_result["segmetrics"] = self.fabric.all_reduce(
                    val_result["segmetrics"]
                )

                if val_dice > self.max_dice:
                    self.max_dice = val_dice.item()
                    isbest = True
                    self.logger.info(
                        f"Max Dice: {self.max_dice:.3f}, Max IoU: {val_iou:.3f}"
                    )
                self.save_checkpoint(epoch, self.max_dice, val_result, isbest)
                self.fabric.barrier()
                isbest = False
            # make log tensorboard
            self.fabric.log_dict(tensorboard_log, epoch)
            self.fabric.log("lr", self.optimizer.param_groups[-1]["lr"], epoch)
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info("Training time {}".format(total_time_str))

    def train_one_epoch(self, data_loader):
        self.model.train()
        self.logger.info(
            f'Current learning rate for different parameter groups: {[it["lr"] for it in self.optimizer.param_groups]}'
        )

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        dice_meter = AverageMeter()
        iou_meter = AverageMeter()
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, batch in enumerate(data_loader):
            loss, outputs, labels = self.step(batch)
            self.fabric.backward(loss)
            if self.config.train.clip_grad:
                grad_norm = self.fabric.clip_gradients(
                    self.model,
                    self.optimizer,
                    max_norm=self.config.train.clip_grad,
                    error_if_nonfinite=False,
                )
            else:
                grad_norm = get_grad_norm(self.model.parameters())
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step_update(self.epoch * num_steps + idx)
            norm_meter.update(grad_norm)
            loss_meter.update(loss.item(), labels.size(0))
            norm_meter.update(grad_norm)
            outputs = outputs.argmax(dim=1)
            dice_meter.update(self.Dice(outputs, labels).item())
            iou_meter.update(self.IOU(outputs, labels).item())
            batch_time.update(time.time() - end)
            end = time.time()
        lr = self.optimizer.param_groups[-1]["lr"]
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        self.logger.info(
            f"Train: [{self.epoch}/{self.config.train.epochs}][{idx}/{num_steps}]\t"
            f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
            f"time {batch_time.avg:.4f} ({batch_time.val:.4f})\t"
            f"loss {loss_meter.avg:.4f} ({loss_meter.val:.4f})\t"
            f"DICE {dice_meter.avg:.3f} ({dice_meter.val:.3f})\t"
            f"IOU {iou_meter.avg:.3f} ({iou_meter.val:.3f})\t"
            f"grad_norm {norm_meter.avg:.4f} ({norm_meter.val:.4f})\t"
            f"mem {memory_used:.0f}MB"
        )
        epoch_time = time.time() - start
        self.logger.info(
            f"EPOCH {self.epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )
        return dice_meter.avg, iou_meter.avg, loss_meter.avg

    @torch.no_grad()
    def validate(self, data_loader):
        self.model.eval()
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        mask_pre = []
        mask_gt = []
        mask_path = []

        end = time.time()
        for idx, batch in enumerate(data_loader):
            loss, outputs, labels = self.step(batch)
            mask_gt.append(labels)
            mask_pre.append(outputs.argmax(dim=1))
            mask_path.append(batch["mask_path"])
            loss_meter.update(loss.item(), labels.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

        mask_gt_all = torch.cat(mask_gt, dim=0)
        mask_pre_all = torch.cat(mask_pre, dim=0)

        mask_path_all = [item for sublist in mask_path for item in sublist]

        segmetrics = get_seg_fromarray(mask_gt_all, mask_pre_all)

        dice = segmetrics["Dice"][0]
        iou = segmetrics["IoU"][0]

        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        self.logger.info(
            f"Test: [{idx}/{len(data_loader)}]\t"
            f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
            f"Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"DICE {dice:.3f}\t"
            f"IOU {iou:.3f}"
            f"Mem {memory_used:.0f}MB"
        )

        val_result = {
            "mask_pre_all": mask_pre_all.cpu(),
            "mask_path_all": mask_path_all,
            "mask_gt_all": mask_gt_all.cpu(),
            "segmetrics": segmetrics,
        }

        return dice, iou, loss, val_result

    @torch.no_grad()
    def test(self):
        if self.config.model.resume:
            self.load_resume()
        else:
            raise ValueError("No checkpoint loaded for testing")
        self.load_resume()
        self.logger.info("Start testing")
        dice, iou, loss, test_result = self.validate(self.dataloader_test)
        self.logger.info(f"Test Dice: {dice:.3f}, Test IoU: {iou:.3f}")
        mask_path_dir = save_seg_pre_gt(self.config.output, test_result, "_test", dice)
        self.logger.info(f"Test result saved in {mask_path_dir}")

    def step(self, batch):
        samples, labels = batch["image"], batch["mask"]
        labels = labels.long()
        extra_features = self.model.module.backbone(samples)
        if hasattr(self.model.module.decode_head, "forward_with_loss"):
            # the [forward_with_loss] function is used for the model with custom training logic
            loss, outputs, labels = self.model.module.decode_head.forward_with_loss(
                extra_features, labels
            )
            return loss, outputs, labels
        else:
            samples, labels = batch["image"], batch["mask"]
            labels = labels.long()

            decoded = self.model.module.decode_head(extra_features)
            if hasattr(self.model.module, "auxiliary_head"):
                auxiliary_logits = self.model.module.auxiliary_head(extra_features)
                auxiliary_outputs = torch.nn.functional.interpolate(
                    auxiliary_logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                auxiliary_outputs = None

            outputs = torch.nn.functional.interpolate(
                decoded, size=labels.shape[-2:], mode="bilinear", align_corners=False
            )
            if auxiliary_outputs is not None:
                loss = 1 * self.criterion(outputs, labels) + 0.4 * self.criterion(
                    auxiliary_outputs, labels
                )
            else:
                loss = self.criterion(outputs, labels)
            return loss, outputs, labels

    def save_checkpoint(self, epoch, max_dice, val_result, isbest=False):
        # save result
        with self.fabric.rank_zero_first():
            checkpoint_name = None
            if isbest:
                self.logger.info(
                    f"Saving the best model with dice {max_dice:.3f} at epoch {epoch}"
                )
                mask_path_dir = save_seg_pre_gt(
                    self.config.output, val_result, epoch, max_dice
                )
                checkpoint_name = save_segmetrics(
                    self.config.output,
                    val_result,
                    epoch,
                    max_dice,
                    mask_path_dir,
                    isbest,
                )
                self.top_k_results_manager.update(mask_path_dir, max_dice)

            if checkpoint_name is None and (epoch == self.config.train.epochs - 1):
                checkpoint_name = f"last{epoch}.pth"

        self.fabric.barrier()
        self.fabric.broadcast(checkpoint_name, 0)
        # best or save_freq or last epoch
        if checkpoint_name is not None:
            self.state.update({"epoch": epoch, "max_dice": max_dice})
            self.fabric.save(
                os.path.join(self.config.output, checkpoint_name), self.state
            )
            self.logger.info(f"Succeed to save checkpoint to {checkpoint_name}")
            self.fabric.barrier()


if __name__ == "__main__":
    SegTrainer()
