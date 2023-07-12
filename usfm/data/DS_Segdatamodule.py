import os
from typing import Any, Dict, Optional, Tuple

import albumentations as A
import torchvision.transforms as T
from albumentations.pytorch import ToTensorV2
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Subset

from usfm.data.components.segdataset import SegBaseDataset, SegVocDataset

train_transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ToFloat(max_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(width=512, height=512),
        A.ToFloat(max_value=255),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1),
        ToTensorV2(),
    ]
)


class DSSegBaseModule(LightningDataModule):
    def __init__(
        self,
        data_path,
        transforms=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters(logger=True, ignore=["transforms"])
        if transforms is None:
            self.train_transforms = train_transforms
            self.val_transforms = val_transforms
        else:
            self.train_transforms = T.Compose(transforms.train)
            self.val_transforms = T.Compose(transforms.val)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: str = None) -> None:
        data_root = self.hparams.data_path.data_root
        data_split = self.hparams.data_path.data_split
        if stage == "fit" or stage is None:
            if data_split.train is not None:
                train_path = os.path.join(data_root, data_split.train)
                self.train_dataset = SegBaseDataset(train_path, self.train_transforms)
            else:
                self.train_dataset = None

            if data_split.val is not None:
                val_path = os.path.join(data_root, data_split.val)
                self.val_dataset = SegBaseDataset(val_path, self.val_transforms)
            else:
                self.val_dataset = None

            if data_split.vis is not None and data_split.val is not None:
                select_index = list(
                    range(
                        0,
                        data_split.vis.samples,
                        1,
                    )
                )
                self.vis_dataset = Subset(self.val_dataset, select_index)
            else:
                self.vis_dataset = None

        if stage == "test":
            if data_split.test is not None:
                test_path = os.path.join(data_root, data_split.test)
                self.test_dataset = SegBaseDataset(test_path, self.val_transforms)
            else:
                self.test_dataset = None

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def vis_dataloader(self):
        return DataLoader(
            dataset=self.vis_dataset,
            batch_size=self.hparams.data_path.data_split.vis.samples,
            num_workers=0,
            pin_memory=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


class DSSegVocModule(DSSegBaseModule):
    def __init__(
        self,
        data_path,
        transforms=None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__(data_path, transforms, batch_size, num_workers, pin_memory, **kwargs)

    def setup(self, stage: str = None) -> None:
        data_root = self.hparams.data_path.data_root
        data_split = self.hparams.data_path.data_split
        image_type = self.hparams.data_path.image_type
        if stage == "fit" or stage is None:
            self.train_dataset = SegVocDataset(
                data_root, data_split.train, image_type, self.train_transforms
            )

            if data_split.val is not None:
                self.val_dataset = SegVocDataset(
                    data_root, data_split.val, image_type, self.train_transforms
                )
            else:
                self.val_dataset = None

            if data_split.vis is not None and data_split.val is not None:
                select_index = list(
                    range(
                        0,
                        data_split.vis.samples,
                        1,
                    )
                )
                self.vis_dataset = Subset(self.val_dataset, select_index)
            else:
                self.vis_dataset = None

        if stage == "test":
            self.train_dataset = SegVocDataset(
                data_root, data_split.test, image_type, self.train_transforms
            )
        else:
            self.test_dataset = None


if __name__ == "__main__":
    from omegaconf import DictConfig

    data_path = {
        "data_root": "data/DownStream/Seg/Thyroid/dataset_folder",
        "data_split": {
            "train": "training_set",
            "val": "test_set",
            "test": "test_set",
            "vis": {"samples": 16, "interval": 10},
        },
    }
    augmentation = {
        "imagenet_default_mean_and_std": False,
        "input_size": 224,
        "patch_size": 16,
        "second_input_size": 112,
        "train_interpolation": "bicubic",
        "second_interpolation": "lanczos",
        "num_mask_patches": 75,
        "min_mask_patches_per_block": 16,
        "max_mask_patches_per_block": None,
        "discrete_vae_type": "dall-e",
    }
    dm = DSSegBaseModule(data_path=DictConfig(data_path), augmentation=DictConfig(augmentation))
    dm.setup()
    print(f"len(train_dataloader): {len(dm.train_dataloader()):d}")
    print(f"len(val_dataloader): {len(dm.val_dataloader()):d}")
    for i, batch in enumerate(dm.train_dataloader()):
        print(
            "index: {:d}, image_shape: {:s}, mask_shape: {:s}".format(
                i, str(batch["image"].shape), str(batch["mask"].shape)
            )
        )
        break
