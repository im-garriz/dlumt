import torch
import numpy as np

from dlumt.typing import *


def segmentation_logits_to_mask(y_hat_logits: FP32_TENSOR) -> FP32_TENSOR:
    """
    Convert segmentation logits to a mask.

    This function converts logits output from a segmentation model to a mask.
    It handles both binary and multi-class segmentation, and can process inputs
    in various tensor shapes: (batch_size, classes, height, width), (classes, height, width)
    and (height, width) for binary segmentation.

    Args:
        y_hat_logits (FP32_TENSOR): The logits tensor from the segmentation model.
                                    Can be a PyTorch tensor or a numpy array.

    Returns:
        FP32_TENSOR: The resulting mask tensor. The type (numpy or PyTorch) matches the input type.

    Raises:
        ValueError: If the input tensor is not 2, 3, or 4-dimensional.
    """
    check_tensor_is_FP32_torch_or_numpy_array(y_hat_logits)

    from_numpy = False
    if isinstance(y_hat_logits, np.ndarray):
        y_hat_logits = torch.tensor(y_hat_logits)
        from_numpy = True

    ndim = y_hat_logits.ndim
    if ndim == 3:
        # Channel dim is the first
        channels_dim = 0
    elif ndim == 4:
        # Channel dim is the second
        channels_dim = 1
    elif ndim == 2:
        # No channel dim -> Adds it
        y_hat_logits = y_hat_logits.unsqueeze(0)
        channels_dim = 0
    else:
        raise ValueError(
            f"The input tensor must be 3-dimensional (NC, H, W), 4-dimensional (BS, NC, H, W) or 2-dimensional (H, W), only for the case of binary segmentation. {ndim} dimensional obtained."
        )

    if y_hat_logits.shape[channels_dim] > 1:
        y_hat = (
            torch.nn.Softmax(dim=channels_dim)(y_hat_logits)
            .argmax(dim=channels_dim)
            .unsqueeze(dim=channels_dim)
            .long()
        )
    else:
        y_hat = torch.zeros_like(y_hat_logits, dtype=torch.long)
        y_hat[y_hat_logits.sigmoid() >= 0.5] = 1

    if ndim == 2:
        y_hat = y_hat[0, ...]

    if from_numpy:
        y_hat = y_hat.numpy().astype(np.long)

    return y_hat
