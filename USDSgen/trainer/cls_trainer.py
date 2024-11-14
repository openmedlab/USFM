import datetime
import os
import pickle
import time

import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from timm.utils import AverageMeter

from usdsgen.utils.logger import array_to_markdown
from usdsgen.utils.modelutils import get_grad_norm

from .basetrainer import BaseTrainer


class ClsTrainer(BaseTrainer):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.max_accuracy = 0.0
        # task specific setting
        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "max_accuracy": self.max_accuracy,
            "scaler": self.scaler,
            "epoch": self.epoch,
            "config": self.config,
        }

        # Verify if the num of classes in training datafolder is same to the config
        assert (
            len(self.dataloader_val.dataset.classes) == self.config.data.num_classes
        ), "The num of classes in training datafolder is not same to the config."
        self.logger.info(f"num of classes: {len(self.dataloader_val.dataset.classes)}")
        self.logger.info(f"class_to_index: {self.dataloader_val.dataset.class_to_idx}")
        # cm row name setting
        self.row_name = list(self.dataloader_val.dataset.class_to_idx.keys())
        # fabric setting datasets & use_distributed_sampler

    def fit(self):
        self.logger.info("Start fiting")
        self.check_resume()
        self.load_resume()

        isbest = False
        start_time = time.time()

        start_epoch = max(self.config.train.start_epoch, self.epoch)
        for epoch in range(start_epoch, self.config.train.epochs):
            self.epoch = epoch
            train_acc, _, train_loss = self.train_one_epoch(self.dataloader_train)
            tensorboard_log = {
                "loss": {
                    "train_loss": train_loss,
                },
                "acc": {
                    "train_acc": train_acc,
                },
            }

            # validation and save the best model
            if epoch % self.config.train.val_freq == 0:
                val_acc, _, val_loss, loginfo = self.validate(self.dataloader_val)
                tensorboard_log["loss"]["val_loss"] = val_loss
                tensorboard_log["acc"]["val_acc"] = val_acc
                if val_acc > self.max_accuracy:
                    self.max_accuracy = val_acc
                    isbest = True
                    self.logger.info(f"Max accuracy: {self.max_accuracy:.3f}")
                    self.logger.info(
                        f"Accuracy of the network on the test images: {self.max_accuracy:.3f}"
                    )
                    self.save_checkpoint(epoch, self.max_accuracy, loginfo, isbest)
            else:
                pass

            if epoch == self.config.train.epochs - 1:
                self.save_checkpoint(epoch, self.max_accuracy, loginfo, isbest)

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            self.logger.info("Training time {}".format(total_time_str))

            # make log tensorboard
            self.fabric.log_dict(tensorboard_log, epoch)
            self.fabric.log("lr", self.optimizer.param_groups[-1]["lr"], epoch)

    def train_one_epoch(self, data_loader):
        self.model.train()
        # self.logger.info(
        #     f'Current learning rate for different parameter groups: {[round(it["lr"], 2) for it in self.optimizer.param_groups]}'
        # )

        num_steps = len(data_loader)
        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        y_t = []
        y_p = []
        norm_meter = AverageMeter()

        start = time.time()
        end = time.time()
        for idx, (samples, labels) in enumerate(data_loader):
            labels = labels.long()
            outputs = self.model(samples)

            # gradient accumulation
            is_accumulating = idx % self.config.train.accumulation_steps != 0
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating):
                loss = self.criterion(outputs, labels)
                # .backward() accumulates when .zero_grad() wasn't called
                self.fabric.backward(loss)

            if not is_accumulating:
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
                if "plateau" not in type(self.lr_scheduler).__name__.lower():
                    self.lr_scheduler.step_update(self.epoch * num_steps + idx)
                norm_meter.update(grad_norm)

            loss_meter.update(loss.item(), labels.size(0))
            norm_meter.update(grad_norm)
            batch_time.update(time.time() - end)
            end = time.time()

            y_t.append(labels.detach())
            y_p.append(outputs.argmax(1).detach())

        y_t = torch.cat(y_t, dim=0)
        y_p = torch.cat(y_p, dim=0)
        y_t_all = self.fabric.all_gather(y_t).reshape(-1).cpu().numpy()
        y_p_all = self.fabric.all_gather(y_p).reshape(-1).cpu().numpy()

        cm = confusion_matrix(y_t_all, y_p_all)
        acc = balanced_accuracy_score(y_t_all, y_p_all)

        if "plateau" in type(self.lr_scheduler).__name__.lower():
            self.lr_scheduler.step(self.epoch, acc)

        lr = self.optimizer.param_groups[-1]["lr"]
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        etas = batch_time.avg * (num_steps - idx)
        self.logger.info(
            f"Train: [{self.epoch}/{self.config.train.epochs}]\t"
            f"eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t"
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"acc {acc:2f}\t"
            f"grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t"
            f"mem {memory_used:.0f}MB"
        )
        epoch_time = time.time() - start
        self.logger.info(
            f"EPOCH {self.epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}"
        )

        self.writer.add_text(
            "cm/train",
            f"#train\n{array_to_markdown(cm, self.row_name, self.row_name)}\n\n",
            self.epoch,
        )
        return acc, cm, loss

    @torch.no_grad()
    def validate(self, data_loader):
        criterion = torch.nn.CrossEntropyLoss()
        self.model.eval()

        batch_time = AverageMeter()
        loss_meter = AverageMeter()
        y_t = []
        outputs_p = []
        y_p = []
        All_feature = []

        end = time.time()
        for idx, (samples, labels) in enumerate(data_loader):
            labels = labels.long()
            feature = self.model.module.forward_features(samples).detach()
            outputs = self.model(samples)
            All_feature.append(feature.detach())
            loss = criterion(outputs, labels)

            loss_meter.update(loss.item(), labels.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            y_t.append(labels.detach())
            outputs_p.append(outputs.detach())
            y_p.append(outputs.argmax(1).detach())

        y_t = torch.cat(y_t, dim=0)
        outputs_p = torch.cat(outputs_p, dim=0)
        y_p = torch.cat(y_p, dim=0)
        all_feature = torch.cat(All_feature, dim=0)

        y_t_all = self.fabric.all_gather(y_t).reshape(-1).cpu().numpy()
        y_p_all = self.fabric.all_gather(y_p).reshape(-1).cpu().numpy()
        outputs_p_all = self.fabric.all_gather(outputs_p).reshape(-1).cpu().numpy()
        all_feature = self.fabric.all_gather(all_feature).flatten(0, 1).cpu().numpy()

        cm = confusion_matrix(y_t_all, y_p_all)
        acc = balanced_accuracy_score(y_t_all, y_p_all)
        memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        self.logger.info(
            f"time {batch_time.val:.4f} ({batch_time.avg:.4f})\t"
            f"loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t"
            f"acc {acc:2f}\t"
            f"mem {memory_used:.0f}MB"
        )

        loginfo = {
            "y_t": y_t_all,
            "y_p": y_p_all,
            "outputs": outputs_p_all,
            "all_feature": all_feature,
        }

        self.writer.add_text(
            "cm/train",
            f"#test\n{array_to_markdown(cm, self.row_name, self.row_name)}",
            self.epoch,
        )
        return acc, cm, loss, loginfo

    @torch.no_grad()
    def test(self):
        if self.config.model.resume:
            self.load_resume()
        else:
            raise ValueError("No checkpoint loaded for testing")
        self.load_resume()
        self.logger.info("Start testing")
        acc, cm, loss, loginfo = self.validate(self.dataloader_test)
        self.logger.info(f"bAccuracy of the network on the test images: {acc:.3f}")
        self.logger.info(f"Confusion matrix: \n{cm}")
        self.logger.info(f"Loss: {loss:.3f}")
        np.savetxt(
            os.path.join(self.config.output, "prediction_result.csv"),
            np.concatenate(
                [loginfo["y_t"].reshape(-1, 1), loginfo["y_p"].reshape(-1, 1)], axis=1
            ),
            delimiter=",",
            fmt="%d",
        )

    def save_checkpoint(self, epoch, max_accuracy, loginfo, isbest=False):
        # best or save_freq or last epoch
        self.fabric.barrier()
        checkpoint_name = "best_ckpt.pth" if isbest else f"ckpt_epoch_{epoch}.pth"
        self.state.update({"epoch": epoch, "max_accuracy": max_accuracy})
        self.fabric.save(os.path.join(self.config.output, checkpoint_name), self.state)
        isbest = False

        with self.fabric.rank_zero_first():
            plot_path_dir = os.path.join(
                self.config.output, f"best{epoch}_acc{max_accuracy:.3f}"
            )
            os.makedirs(plot_path_dir, exist_ok=True)

        with open(os.path.join(plot_path_dir, "loginfo.pkl"), "wb") as f:
            pickle.dump(loginfo, f)


if __name__ == "__main__":
    ClsTrainer()
