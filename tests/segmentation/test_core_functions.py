import pytest
import numpy as np
import torch

from dlumt.segmentation.core import segmentation_logits_to_mask
from dlumt.typing import *


class TestSegmentationLogitsToMask:

    @pytest.mark.parametrize(
        "x",
        [
            (np.random.rand(3, 34, 34).astype(np.float64)),
            (np.random.rand(3, 34, 34).astype(np.long)),
            (torch.rand((3, 34, 34), dtype=torch.double)),
        ],
    )
    def test_raises_exception_when_tensor_not_FP32(self, x: FP32_TENSOR) -> None:
        with pytest.raises(AssertionError):
            _ = segmentation_logits_to_mask(x)

    @pytest.mark.parametrize(
        "x",
        [
            (torch.rand((3, 34, 34, 34, 34), dtype=torch.float)),
            (torch.rand((34), dtype=torch.float)),
            (torch.rand((3, 34, 34, 34, 34, 2), dtype=torch.float)),
        ],
    )
    def test_raises_exception_when_input_not_2_3_or_4_dimensional(
        self, x: FP32_TENSOR
    ) -> None:
        with pytest.raises(AssertionError):
            _ = segmentation_logits_to_mask(x)


    # def test_binary_segmentation(self, logits: FP32_TENSOR, classes: INT64_TENSOR) -> None:

    # #     logits = np.random.rand()
