import os
import random

import numpy as np
import torch
import torch.fft as fft
from PIL import Image
from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)
from typing_extensions import Tuple


def DeNormalize(tensor, mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD):
    """DeNormalize a tensor image with mean and standard deviation.

    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    Returns:
        Tensor: DeNormalized image.
    """
    if tensor.dim() == 3:
        assert tensor.dim() == 3, "Expected image [CxHxW]"
        assert tensor.size(0) == 3, "Expected RGB image [3xHxW]"
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)

    elif tensor.dim() == 4:
        # batch mode
        if tensor.size(1) == 1:
            tensor = torch.cat([tensor, tensor, tensor], dim=1)
        for t, m, s in zip((0, 1, 2), mean, std):
            tensor[:, t, :, :].mul_(s).add_(m)

        return tensor


class MaskGenerator:
    def __init__(
        self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        return mask


class Frequency_MASK(torch.nn.Module):
    """
    Band-stop filter (annular mask in frequency domain)
    """

    def __init__(self, image_size, FMASK_config):
        super().__init__()
        self.image_size = image_size
        self.f_mask_ratio = FMASK_config.Ratio
        f_mask_group = self.get_f_mask_group(image_size, FMASK_config)
        self.mask_index = list(range(f_mask_group.shape[0]))
        self.num_mask = int(self.f_mask_ratio * len(self.mask_index))
        self.register_buffer("f_mask_group", f_mask_group)

    @staticmethod
    def get_f_mask_group(image_size, FMASK_config):
        shape_x, shape_y = image_size, image_size
        increase_factor = FMASK_config.IncreaseFactor
        max_increase = FMASK_config.MaxIncrease
        f_mask_group = []
        base_BW = FMASK_config.BaseBW
        center_protect = FMASK_config.CenterProtect
        center_x, center_y = int(shape_x / 2), int(shape_y / 2)
        start_x, start_y = center_x - center_protect, center_y - center_protect
        end_x, end_y = center_x + center_protect, center_y + center_protect
        BW = base_BW
        while start_x >= 0 and start_y >= 0:
            rap = torch.ones((shape_x, shape_y))
            for x in range(shape_x):
                for y in range(shape_y):
                    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                    if end_x - center_x - BW <= distance <= end_x - center_x:
                        rap[x, y] = 0

            # rap[start_x:end_x, start_y:end_y] = 0
            f_mask_group.append(rap)
            start_x, start_y = center_x - BW, center_y - BW
            end_x, end_y = center_x + BW, center_y + BW
            BW_increase = int(BW * increase_factor)
            if BW_increase > max_increase:
                BW_increase = max_increase
            BW = BW + BW_increase

        rap_mask = [
            f_mask_group[i + 1] - f_mask_group[i] + 1
            for i in range(len(f_mask_group) - 1)
        ]

        return torch.stack(rap_mask, dim=0)

    def forward(self, img):
        # get rap mask combination
        random.shuffle(self.mask_index)
        combinated_mask = torch.ones_like(self.f_mask_group[0])
        for i in range(self.num_mask):
            combinated_mask *= self.f_mask_group[self.mask_index[i]]

        # Fourier transform
        img_tensor = torch.Tensor(img)
        fre = fft.fft2(img_tensor, dim=(1, 2))
        fre_shift = fft.fftshift(fre)
        fre_ishift = fft.ifftshift(fre_shift * combinated_mask)
        iimg = fft.ifft2(fre_ishift)
        iimg = torch.abs(iimg)
        return iimg, combinated_mask.unsqueeze(0)


class BlockwiseMaskGenerator:
    """Generate random block for the image.

    Args:
        input_size (int): Size of input image. Defaults to 192.
        mask_patch_size (int): Size of each block mask. Defaults to 32.
        model_patch_size (int): Patch size of each token. Defaults to 4.
        mask_ratio (float): The mask ratio of image. Defaults to 0.6.
        mask_color (str): Filling color of the MIM mask in {'mean', 'zero'}.
            Defaults to 'zero'.
    """

    def __init__(
        self,
        input_size=192,
        mask_patch_size=32,
        model_patch_size=4,
        mask_ratio=0.6,
        mask_only=False,
        mask_color="zero",
    ):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        self.mask_only = mask_only
        self.mask_color = mask_color
        assert self.mask_color in [
            "mean",
            "zero",
            "rand",
        ]
        if self.mask_color != "zero":
            assert mask_only is False

        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0

        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size**2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self, img: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_idx = np.random.permutation(self.token_count)[: self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        mask = torch.from_numpy(mask)  # [H, W]

        if self.mask_color == "mean":
            mask_ = mask.clone()
            mask_ = (
                mask_.repeat_interleave(self.model_patch_size, 0)
                .repeat_interleave(self.model_patch_size, 1)
                .contiguous()
            )
            img = img.clone()
            mean = img.mean(dim=[1, 2])
            for i in range(img.size(0)):
                img[i, mask_ == 1] = mean[i]

        if self.mask_only:
            return mask
        else:
            return img, mask
