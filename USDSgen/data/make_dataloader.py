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

    # 数据集拆分
    org_len = len(dataset_train)
    if config.data.train_ratio is not None and config.data.train_ratio < 1:
        # Stratified Sampling from dataset_train with the train_ratio
        train_class = dataset_train.targets
        print(train_class)
        from sklearn.model_selection import train_test_split

        stratify = train_class if "cls" in config.data.type else None
        train_index, _ = train_test_split(
            list(range(org_len)),
            train_size=config.data.train_ratio,
            random_state=42,
            stratify=stratify,
        )
        dataset_train = Subset(dataset_train, train_index)

    logger.info(
        f"Finally build dataset: train images = {len(dataset_train)}/{org_len}({config.data.train_ratio}), val images = {len(dataset_val)}, test images = {len(dataset_test)}"
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
