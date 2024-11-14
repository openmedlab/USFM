import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from .datasets import build_cls_dataset, build_seg_dataset


def build_loader(config, logger):
    if "cls" in config.data.type.lower():
        dataset_train, dataset_val, dataset_test = build_cls_dataset(config, logger)
    elif "seg" in config.data.type.lower():
        dataset_train, dataset_val, dataset_test = build_seg_dataset(config, logger)
    else:
        raise NotImplementedError("We only support seg and cls now.")

    logger.info(
        f"Finally build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}, test images = {len(dataset_test)}"
    )

    # dataloader will be setup by fabric, so no distribution sample here
    dataloader_train = DataLoader(
        dataset_train,
        # sampler=sampler_train,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        dataset_val,
        # sampler=sampler_val,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )

    dataloader_test = DataLoader(
        dataset_test,
        # sampler=sampler_test,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        drop_last=False,
    )

    return dataloader_train, dataloader_val, dataloader_test
