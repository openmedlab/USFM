from timm.data.constants import (
    IMAGENET_DEFAULT_MEAN,
    IMAGENET_DEFAULT_STD,
    IMAGENET_INCEPTION_MEAN,
    IMAGENET_INCEPTION_STD,
)


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
        assert tensor.size(1) == 3, "Expected RGB image [3xHxW]"
        for t, m, s in zip((0, 1, 2), mean, std):
            tensor[:, t, :, :].mul_(s).add_(m)

        return tensor
