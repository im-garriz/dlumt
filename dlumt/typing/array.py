from typing import Union
import numpy as np
import numpy.typing as npt
import torch


__all__ = ["FP32_TENSOR", "check_input_is_torch_or_numpy_array"]


FP32_TENSOR = Union[npt.NDArray[np.float32], torch.Tensor]
"""
A type alias representing either a NumPy array of dtype float32 or a PyTorch tensor.
"""

def check_input_is_torch_or_numpy_array(x):
    """
    Check if the input is a NumPy array or a PyTorch tensor. If not,
    raises a TypeError.

    Args:
        x: Input object to be checked.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    if not isinstance(x, np.ndarray) and not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Input must be a NumPy array or a PyTorch tensor. Got {type(x)}."
        )
