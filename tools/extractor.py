"""
Author: Zehui Lin
Date: 2024-12-11 16:09:53
LastEditors: Zehui Lin
LastEditTime: 2024-12-11 16:19:54
FilePath: /USFM/tools/extractor.py
Description: 写点东西描述一下这个文件叭~

Using USFM as a feature extractor

"""

import os

import cv2
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


def prepare_inputs():

    dataset_path = "datasets/Seg/toy_dataset/training_set/image"
    dataset_list = os.listdir(dataset_path)

    input_list = []
    bar = tqdm(total=len(dataset_list), ncols=100, desc="Prepare Inputs")
    for filename in dataset_list:
        bar.update(1)

        image = cv2.imread(os.path.join(dataset_path, filename))
        image = np.mean(image, axis=2)

        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = (image - image.mean()) / image.std()
        image_data = np.repeat(image[:, :, None], 3, axis=-1)

        input_list.append(image_data)

    return input_list


def main():

    import logging

    import yaml
    from omegaconf import OmegaConf

    from usdsgen.modules.backbone.vision_transformer import build_vit

    with open("configs/model/Cls/vit.yaml") as f:
        cfg = OmegaConf.create(yaml.load(f, Loader=yaml.FullLoader))
    cfg.model.model_cfg.num_classes = 6
    cfg.model.model_cfg.backbone.pretrained = "./assets/FMweight/USFM_latest.pth"

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())

    model = build_vit(cfg.model.model_cfg, logger)
    model.to("cuda:0")
    model.eval()

    input_list = prepare_inputs()
    feature_list = []
    bar = tqdm(total=len(input_list), ncols=100, desc="Extract Features")
    for image in input_list:
        bar.update(1)
        x = Variable(torch.from_numpy(image).float().cuda())
        x = x.permute(2, 0, 1).unsqueeze(0)
        feature = model.forward_features(x)  # 1, 768
        feature = feature.cpu().data.numpy().flatten()
        feature_list.append(feature)

    feature_list = np.array(feature_list)
    np.save(
        "tools/saved_feature.npy",
        feature_list,
    )


if __name__ == "__main__":
    main()
