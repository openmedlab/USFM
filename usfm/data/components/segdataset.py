import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class SegBaseDataset(Dataset):
    def __init__(self, folder, transforms=None):
        super().__init__()
        self.update_datalist(folder)
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = self.image_list[index]
        mask_file = self.mask_list[index]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = mask / np.max(mask)
        if self.transforms is not None:
            image_mask = self.transforms(image=image, mask=mask)
        return image_mask

    def update_datalist(self, folder):
        image_path = os.path.join(folder, "image")
        mask_path = os.path.join(folder, "mask")
        filenames = []
        for root, dirs, files in os.walk(image_path):
            for file in files:
                filenames.append(os.path.join(root, file))

        self.image_list = filenames
        self.mask_list = [i.replace("set/image", "set/mask") for i in filenames]

    def __len__(self):
        return len(self.image_list)


class SegVocDataset(Dataset):
    def __init__(self, folder, data_type="train", image_type="png", transforms=None):
        super().__init__()
        self.update_datalist(folder, data_type, image_type)
        self.transforms = transforms

    def __getitem__(self, index):
        image_file = self.image_list[index]
        mask_file = self.mask_list[index]
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
        mask = mask / np.max(mask)
        if self.transforms is not None:
            image_mask = self.transforms(image=image, mask=mask)
        return image_mask

    def update_datalist(self, folder, data_type, image_type):
        filenames = np.loadtxt(os.path.join(folder, "ImageSets", data_type + ".txt"), dtype=str)
        image_filenames = [i + "." + image_type for i in filenames]
        mask_filenames = [i + ".png" for i in filenames]
        self.image_list = [os.path.join(folder, "JPEGImages", i) for i in image_filenames]
        self.mask_list = [os.path.join(folder, "SegmentationClass", i) for i in mask_filenames]

    def __len__(self):
        return len(self.image_list)
