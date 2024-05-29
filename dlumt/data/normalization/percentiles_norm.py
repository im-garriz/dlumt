from typing import Tuple, Optional
import numpy as np
import torch

from dlumt.typing.array import *


class PercentilesNormalizer:
    """
    A class for normalizing and denormalizing arrays (torch or numpy) using percentiles.

    Attributes:
        min_percentiles (np.ndarray): Minimum percentiles for normalization.
        max_percentiles (np.ndarray): Maximum percentiles for normalization.
        n_channels (int): Number of channels in the input arrays.
    """

    def __init__(self, min_percentiles: np.ndarray, max_percentiles: np.ndarray):
        """
        Initialize PercentilesNormalizer with min and max percentiles. They must be one-dimensional,
        both with the same number of elements and of type FP32. Examples:

        min_percentiles -> np.array([5.0, 4.0, 8.0], dtype=np.float32)
        max_percentiles -> np.array([245.0, 224.0, 234.0], dtype=np.float32)

        Args:
            min_percentiles (np.ndarray): Minimum percentiles for normalization per channel.
            max_percentiles (np.ndarray): Maximum percentiles for normalization per channel.

        Raises:
            AssertionError: If min_percentiles or max_percentiles have incorrect dtype or shape.
        """

        assert min_percentiles.dtype == np.float32
        assert max_percentiles.dtype == np.float32
        assert min_percentiles.ndim == 1
        assert max_percentiles.ndim == 1
        assert min_percentiles.shape == max_percentiles.shape

        self.min_percentiles = min_percentiles
        self.max_percentiles = max_percentiles

        self.n_channels = self.min_percentiles.shape[0]

    @staticmethod
    def check_input(
        in_array: FP32_TENSOR,
    ) -> Tuple[bool, Optional[torch.device]]:
        """
        Check the input array to be either numpy array or torch tensor. If it is a torch tensor,
        it returns the torch.device to be casted afterwards.

        Args:
            in_array (FP32_TENSOR): Input array to be checked and casted.

        Returns:
            Tuple[bool, Optional[torch.device]]: Tuple containing a boolean indicating
            if the input array is a torch tensor, and optionally the device if the input
            array is a torch tensor.
        """
        check_input_is_torch_or_numpy_array(in_array)
        is_torch_tensor = False
        device = None
        if isinstance(in_array, torch.Tensor):
            is_torch_tensor = True
            device = in_array.device

        return is_torch_tensor, device

    def normalize_NC_H_W(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Normalize a 3D input array (NC, H, W) along the channel dimension.

        Args:
            in_array (FP32_TENSOR): Input array to be normalized.

        Returns:
            FP32_TENSOR: Normalized array.
        """
        clipped_array = np.clip(
            in_array,
            self.min_percentiles[:, None, None],
            self.max_percentiles[:, None, None],
        )
        normalized_array = (clipped_array - self.min_percentiles[:, None, None]) / (
            self.max_percentiles[:, None, None] - self.min_percentiles[:, None, None]
        )

        return normalized_array

    def normalize_BS_NC_H_W(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Normalize a 4D input array (BS, NC, H, W) along the channel dimension.

        Args:
            in_array (FP32_TENSOR): Input array to be normalized.

        Returns:
            FP32_TENSOR: Normalized array.
        """
        clipped_array = np.clip(
            in_array,
            self.min_percentiles[None, :, None, None],
            self.max_percentiles[None, :, None, None],
        )
        normalized_array = (
            clipped_array - self.min_percentiles[None, :, None, None]
        ) / (
            self.max_percentiles[None, :, None, None]
            - self.min_percentiles[None, :, None, None]
        )

        return normalized_array

    def denormalize_NC_H_W(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Denormalize a 3D input array (NC, H, W) along the channel dimension.

        Args:
            in_array (FP32_TENSOR): Input array to be denormalized.

        Returns:
            FP32_TENSOR: Denormalized array.
        """
        denormalized_array = (
            in_array
            * (
                self.max_percentiles[:, None, None]
                - self.min_percentiles[:, None, None]
            )
            + self.min_percentiles[:, None, None]
        )

        return denormalized_array

    def denormalize_BS_NC_H_W(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Denormalize a 4D input array (BS, NC, H, W) along the channel dimensions.

        Args:
            in_array (FP32_TENSOR): Input array to be denormalized.

        Returns:
            FP32_TENSOR: Denormalized array.
        """
        denormalized_array = (
            in_array
            * (
                self.max_percentiles[None, :, None, None]
                - self.min_percentiles[None, :, None, None]
            )
            + self.min_percentiles[None, :, None, None]
        )

        return denormalized_array

    def normalize(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Normalize an input array using percentiles.

        Args:
            in_array (FP32_TENSOR): Input array to be normalized.

        Returns:
            FP32_TENSOR: Normalized array.

        Raises:
            ValueError: If the input array dimension is neither 3 nor 4.
        """

        is_torch_tensor, device = self.check_input(in_array)
        if is_torch_tensor:
            in_array = in_array.detach().cpu().numpy()

        if in_array.ndim == 3:  # NC_H_W
            normalized_array = self.normalize_NC_H_W(in_array)
        elif in_array.ndim == 4:  # BS_NC_H_W
            normalized_array = self.normalize_BS_NC_H_W(in_array)
        else:
            raise ValueError(
                f"in_array must be 3 dimentional (NC, H, W) or 4 dimensional (BS, NC, H, W), got {in_array.ndim} dimensional"
            )

        if is_torch_tensor:
            normalized_array = torch.tensor(normalized_array).to(device)

        return normalized_array  # 0-1

    def denormalize(self, in_array: FP32_TENSOR) -> FP32_TENSOR:
        """
        Denormalize an input array using percentiles.

        Args:
            in_array (FP32_TENSOR): Input array to be denormalized.

        Returns:
            FP32_TENSOR: Denormalized array.

        Raises:
            ValueError: If the input array dimension is neither 3 nor 4.
        """
        is_torch_tensor, device = self.check_input(in_array)
        if is_torch_tensor:
            in_array = in_array.detach().cpu().numpy()

        if in_array.ndim == 3:  # NC_H_W
            denormalized_array = self.denormalize_NC_H_W(in_array)
        elif in_array.ndim == 4:  # BS_NC_H_W
            denormalized_array = self.denormalize_BS_NC_H_W(in_array)
        else:
            raise ValueError(
                f"in_array must be 3 dimentional (NC, H, W) or 4 dimensional (BS, NC, H, W), got {in_array.ndim} dimensional"
            )

        if is_torch_tensor:
            denormalized_array = torch.tensor(denormalized_array).to(device)

        return denormalized_array
