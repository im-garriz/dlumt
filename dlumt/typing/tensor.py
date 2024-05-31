from typing import Union
import numpy as np
import numpy.typing as npt
import torch


__all__ = [
    "FP32_TENSOR",
    "check_tensor_is_FP32_torch_or_numpy_array",
    "INT64_TENSOR",
    "check_tensor_is_INT64_torch_or_numpy_array",
]


TENSOR = Union[np.array, torch.Tensor]
"""
A type alias representing either a NumPy array or a PyTorch tensor.
"""

def check_input_is_numpy_or_torch_tensor(x: TENSOR) -> None:
    """
    Check if the input is either a NumPy array or a PyTorch tensor.

    Args:
        x (TENSOR): Input object to be checked.

    Raises:
        TypeError: If the input is neither a NumPy array nor a PyTorch tensor.
    """
    if not isinstance(x, np.ndarray) and not isinstance(x, torch.Tensor):
        raise TypeError(
            f"Input must be a NumPy array or a PyTorch tensor. Got {type(x)}."
        )


def check_type_torch(x: TENSOR, target_type: str) -> None:
    """
    Check if the PyTorch tensor is of the specified type.

    Args:
        x (TENSOR): Input tensor to be checked.
        target_type (str): The expected type of the PyTorch tensor.

    Raises:
        TypeError: If the input tensor is not of the expected type.
    """
    if isinstance(x, torch.Tensor) and x.type() != target_type:
        raise TypeError(f"Input must be of type {target_type}. Got {x.type()}.")


def check_type_numpy(x: TENSOR, target_type: np.dtype) -> None:
    """
    Check if the NumPy array is of the specified dtype.

    Args:
        x (TENSOR): Input array to be checked.
        target_type (np.dtype): The expected dtype of the NumPy array.

    Raises:
        TypeError: If the input array is not of the expected dtype.
    """
    if isinstance(x, np.ndarray) and x.dtype != target_type:
        raise TypeError(f"Input must be of type {target_type}. Got {x.dtype}.")


FP32_TENSOR = Union[
    npt.NDArray[np.float32], Union[torch.FloatTensor, torch.cuda.FloatTensor]
]
"""
A type alias representing either a NumPy array of dtype float32 or a PyTorch FloatTensor or cuda.FloatTensor.
"""

def check_tensor_is_FP32_torch_or_numpy_array(x: FP32_TENSOR) -> None:
    """
    Check if the input is a float32 NumPy array or a PyTorch FloatTensor.

    Args:
        x (FP32_TENSOR): Input tensor to be checked.

    Raises:
        TypeError: If the input is neither a float32 NumPy array nor a PyTorch FloatTensor.
    """
    check_input_is_numpy_or_torch_tensor(x)
    check_type_torch(x, torch.float)
    check_type_numpy(x, np.float32)


INT64_TENSOR = Union[
    npt.NDArray[np.int64], Union[torch.LongTensor, torch.cuda.LongTensor]
]
"""
A type alias representing either a NumPy array of dtype int64 or a PyTorch LongTensor or cuda.LongTensor.
"""

def check_tensor_is_INT64_torch_or_numpy_array(x: INT64_TENSOR) -> None:
    """
    Check if the input is an int64 NumPy array or a PyTorch LongTensor.

    Args:
        x (INT64_TENSOR): Input tensor to be checked.

    Raises:
        TypeError: If the input is neither an int64 NumPy array nor a PyTorch LongTensor.
    """
    check_input_is_numpy_or_torch_tensor(x)
    check_type_torch(x, torch.long)
    check_type_numpy(x, np.int64)
