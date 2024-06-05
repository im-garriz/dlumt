from typing import Tuple, List
import numpy as np
import pytest
import torch

from dlumt.data import get_patches_origins_from_tile_shape, apply_dihedral_transform
from dlumt.typing import *


class TestGetPatchesOriginsFromTileShape:

    @pytest.mark.parametrize(
        "tile_height,tile_width,patch_size,stride,target",
        [
            (256, 256, 256, 256, [(0, 0)]),
            (256, 256, 256, 128, [(0, 0)]),
            (512, 512, 256, 256, [(0, 0), (0, 256), (256, 0), (256, 256)]),
            (
                550,
                550,
                256,
                256,
                [
                    (0, 0),
                    (0, 256),
                    (0, 294),
                    (256, 0),
                    (256, 256),
                    (256, 294),
                    (294, 0),
                    (294, 256),
                    (294, 294),
                ],
            ),
        ],
    )
    def test_correct_patches_are_generated(
        self,
        tile_height: int,
        tile_width: int,
        patch_size: int,
        stride: int,
        target: List[Tuple[int, int]],
    ) -> None:

        generated_origins = get_patches_origins_from_tile_shape(
            tile_height, tile_width, patch_size, stride
        )

        assert generated_origins == target

    def test_assertions_errors_are_correctly_raised(self) -> None:

        with pytest.raises(AssertionError):
            get_patches_origins_from_tile_shape(512, 512, 256, 257)

        with pytest.raises(AssertionError):
            get_patches_origins_from_tile_shape(51, 512, 256, 256)

        with pytest.raises(AssertionError):
            get_patches_origins_from_tile_shape(512, 51, 256, 256)


class TestApplyDihedralTransforms:

    def test_idx_0_gives_same_tensor(self) -> None:
        x = np.random.rand(1, 32, 32).astype(np.float32)
        x_dihedral = apply_dihedral_transform(x, 0)

        assert np.alltrue(x == x_dihedral)

        x = torch.rand(1, 32, 32)
        x_dihedral = apply_dihedral_transform(x, 0)

        assert torch.all(x == x_dihedral)

    def test_idx_1_works(self) -> None:
        x = np.random.rand(1, 32, 32).astype(np.float32)
        x_dihedral = apply_dihedral_transform(x, 1)

        assert np.alltrue(np.flip(x, -1) == x_dihedral)

        x = torch.rand(1, 32, 32)
        x_dihedral = apply_dihedral_transform(x, 1)

        assert torch.all(x.flip(-1) == x_dihedral)

    def test_idx_1_works_with_4D(self) -> None:
        x = np.random.rand(4, 1, 32, 32).astype(np.float32)
        x_dihedral = apply_dihedral_transform(x, 1)

        assert np.alltrue(np.flip(x, -1) == x_dihedral)

        x = torch.rand(4, 1, 32, 32)
        x_dihedral = apply_dihedral_transform(x, 1)

        assert torch.all(x.flip(-1) == x_dihedral)
