import os

import lightning as L
import torch
import torch.amp
from lightning.fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf
from timm.loss import LabelSmoothingCrossEntropy

from usdsgen.data import build_loader
from usdsgen.models import build_model
from usdsgen.utils.file_manager import auto_resume_helper
from usdsgen.utils.logger import create_logger
from usdsgen.utils.lr_scheduler import build_scheduler
from usdsgen.utils.optimizer import build_optimizer


class BaseTrainer:
    def __init__(self, config: DictConfig) -> None:
        # base setting
        self.config = config
        self.fabric = L.Fabric(
            accelerator=config.L.accelerator,
            strategy=config.L.strategy,
            devices=config.L.devices,
            precision=config.L.precision,
            loggers=TensorBoardLogger(config.output, "logs"),
        )
        self.fabric.launch()
        self.fabric.seed_everything(42)
        # make dir and logger
        os.makedirs(config.output, exist_ok=True)
        self.logger = create_logger(
            output_dir=config.output,
            dist_rank=self.fabric.global_rank,
            name=config.task_name,
        )

        self.writer = self.fabric.loggers[0].experiment
        try:
            # for torch version >=2.4
            self.scaler = torch.amp.GradScaler("cuda")
        except:
            self.scaler = torch.cuda.amp.GradScaler()

        self.epoch = 0

        self.make_dataloader()
        self.make_model()
        self.make_loss()
        self.make_optimizer()

        # model fabric setting up
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)
        self.fabric.loggers[0].log_hyperparams(self.config)

        self.state = {
            "model": self.model,
            "optimizer": self.optimizer,
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "scaler": self.scaler,
            "epoch": self.epoch,
            "config": self.config,
        }

    def fit(self):
        raise NotImplementedError

    def train_one_epoch(self, data_loader):
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, data_loader):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def make_dataloader(self):
        # dataloader setting
        (
            dataloader_train,
            dataloader_val,
            dataloader_test,
        ) = build_loader(self.config, self.logger)
        self.dataloader_train, self.dataloader_val, self.dataloader_test = (
            self.fabric.setup_dataloaders(
                dataloader_train,
                dataloader_val,
                dataloader_test,
                use_distributed_sampler=self.config.L.use_distributed_sampler,
            )
        )

    def make_model(self):
        self.model = build_model(self.config, self.logger)
        n_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        self.logger.info(
            f"Creating model:{self.config.model.model_name}/{OmegaConf.to_container(self.config.model.model_cfg, resolve=True)} for task: {self.config.task}"
        )
        self.logger.info(f"number of params: {n_parameters}")
        if hasattr(self.model, "flops"):
            flops = self.model.flops()
            self.logger.info(f"number of GFLOPs: {flops / 1e9}")

    def make_optimizer(self):
        # lr setting
        linear_scaled_lr = (
            self.config.train.base_lr
            * self.config.data.batch_size
            * self.fabric.world_size
            / 512.0
        )
        linear_scaled_warmup_lr = (
            self.config.train.warmup_lr
            * self.config.data.batch_size
            * self.fabric.world_size
            / 512.0
        )
        linear_scaled_min_lr = (
            self.config.train.min_lr
            * self.config.data.batch_size
            * self.fabric.world_size
            / 512.0
        )
        # gradient accumulation also need to scale the learning rate
        if self.config.train.accumulation_steps > 1:
            linear_scaled_lr = linear_scaled_lr * self.config.train.accumulation_steps
            linear_scaled_warmup_lr = (
                linear_scaled_warmup_lr * self.config.train.accumulation_steps
            )
            linear_scaled_min_lr = (
                linear_scaled_min_lr * self.config.train.accumulation_steps
            )

        self.config.train.base_lr = linear_scaled_lr
        self.config.train.warmup_lr = linear_scaled_warmup_lr
        self.config.train.min_lr = linear_scaled_min_lr

        # optimizer
        self.optimizer = build_optimizer(self.config, self.model, self.logger)
        self.lr_scheduler = build_scheduler(
            self.config, self.optimizer, len(self.dataloader_train)
        )

    def make_loss(self):
        # loss function
        if self.config.train.label_smoothing > 0.0:
            self.criterion = LabelSmoothingCrossEntropy(
                smoothing=self.config.train.label_smoothing
            )
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def check_resume(self):
        # total parameters of model
        # load resume checkpoint or pretrained model
        self.epoch = 0
        # auto_resume
        if self.config.model.resume is None and self.config.train.auto_resume:
            checkpoint_check_path = os.path.dirname(os.path.dirname(self.config.output))
            resume_file = auto_resume_helper(checkpoint_check_path, self.logger)
            if resume_file:
                if self.config.model.resume:
                    self.logger.warning(
                        f"auto-resume changing resume file from {self.config.model.resume} to {resume_file}"
                    )
                self.config.model.resume = resume_file
                self.logger.info(f"auto resuming from {resume_file}")
            else:
                self.logger.info(
                    f"no checkpoint found in {checkpoint_check_path}, ignoring auto resume"
                )

    def load_resume(self):
        if self.config.model.resume is None:
            self.logger.info("No model resume file, can't resume")
            return

        if self.config.train.only_resume_model:
            not_autoload = self.fabric.load(self.config.model.resume)
            self.model.load_state_dict(not_autoload["model"])
            self.epoch = not_autoload["epoch"]
            self.config.train.epoch = self.epoch + self.config.train.epochs
            self.logger.info(
                f"Only resume model from from {self.config.model.resume} at epoch {self.epoch}; The start epoch is {self.epoch} and the total epoch has updated to {self.config.train.epochs}"
            )
        else:
            not_autoload = self.fabric.load(self.config.model.resume, self.state)
            self.epoch = self.state["epoch"]
            self.logger.info(
                f"Resume all states from {self.config.model.resume} at epoch {self.epoch}"
            )

    def save_checkpoint(self, epoch, max_accuracy, loginfo, isbest=False):
        raise NotImplementedError


if __name__ == "__main__":
    BaseTrainer()
