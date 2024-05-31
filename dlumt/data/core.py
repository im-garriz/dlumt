from itertools import product
from typing import List, Tuple
import numpy as np
import torch

from dlumt.typing import *


def get_patches_origins_from_tile_shape(
    tile_height: int, tile_width: int, patch_size: int, stride: int
) -> List[Tuple[int, int]]:
    """
    Calculate the origins of patches from a tile shape.

    This function computes the coordinates of the top-left corners of patches
    within a tile of specified dimensions. The patches are generated based on
    the given patch size and stride.

    Args:
        tile_height (int): Height of the tile.
        tile_width (int): Width of the tile.
        patch_size (int): Size of each patch (assumed square).
        stride (int): Stride between patches.

    Returns:
        List[Tuple[int, int]]: A list of (y, x) coordinates for the top-left
                               corners of the patches.

    Raises:
        AssertionError: If the patch size is smaller than the stride or if the
                       tile dimensions are smaller than the patch size.
    """
    assert (
        patch_size >= stride
    ), f"patch_size must be larger than the stride. Got patch_size: {patch_size}, stride: {stride}"
    assert (
        tile_height >= patch_size
    ), f"tile_height must be larger than patch_size. Got tile_height: {tile_height}, patch_size: {patch_size}"
    assert (
        tile_width >= patch_size
    ), f"tile_width must be larger than patch_size. Got tile_width: {tile_width}, patch_size: {patch_size}"

    x_origs_w = np.arange(0, max(tile_width - patch_size, 1), stride).tolist()
    y_origs_h = np.arange(0, max(tile_height - patch_size, 1), stride).tolist()

    if len(x_origs_w) == 0 or len(y_origs_h) == 0:
        return []

    residual_h, residual_w = tile_height - y_origs_h[-1], tile_width - x_origs_w[-1]
    if residual_h > patch_size:
        y_origs_h.append(tile_height - patch_size)
    if residual_w > patch_size:
        x_origs_w.append(tile_width - patch_size)

    return list(product(y_origs_h, x_origs_w))


def apply_dihedral_transform(x: FP32_TENSOR, transformation_idx: int) -> FP32_TENSOR:
    """
    Apply a dihedral transformation to a tensor.

    This function applies a dihedral transformation (combination of reflections
    and rotations) to the input tensor based on the specified transformation index.

    Args:
        x (FP32_TENSOR): The input tensor to be transformed. Can be a numpy array
                         or a PyTorch tensor.
        transformation_idx (int): The index of the transformation to apply (0-7).

    Returns:
        FP32_TENSOR: The transformed tensor, maintaining the same type as the input.

    Raises:
        AssertionError: If the input is not a valid numpy array or PyTorch tensor.
    """
    check_tensor_is_FP32_torch_or_numpy_array(x)
    from_numpy = False
    if isinstance(x, np.ndarray):
        from_numpy = True
        x = torch.tensor(x)

    if transformation_idx in [1, 3, 4, 7]:
        x = x.flip(-1)
    if transformation_idx in [2, 4, 5, 7]:
        x = x.flip(-2)
    if transformation_idx in [3, 5, 6, 7]:
        x = x.transpose(-1, -2)

    if from_numpy:
        x = x.numpy()

    return x
