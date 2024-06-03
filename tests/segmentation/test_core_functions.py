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

    @pytest.mark.parametrize(
        "logits",
        [
            (3*(torch.rand((25, 25), dtype=torch.float32) - 0.5)),
            (3*(torch.rand((1, 25, 25), dtype=torch.float32) - 0.5)),
            (3*(torch.rand((12, 1, 25, 25), dtype=torch.float32) - 0.5)),
        ],
    )
    def test_binary_segmentation_torch(self, logits: FP32_TENSOR) -> None:
    
        target_mask = (logits.sigmoid() >= 0.5).long()
        generated_mask = segmentation_logits_to_mask(logits)
        assert torch.all(target_mask == generated_mask)
        assert generated_mask.min() == 0
        assert generated_mask.max() == 1
        assert target_mask.shape == generated_mask.shape
        assert target_mask.type()== generated_mask.type()
        
        
    @pytest.mark.parametrize(
        "logits",
        [
            (3*(torch.rand((4, 25, 25), dtype=torch.float32) - 0.5)),
            (3*(torch.rand((12, 6, 25, 25), dtype=torch.float32) - 0.5)),
        ],
    )
    def test_multiclass_segmentation_torch(self, logits: FP32_TENSOR) -> None:
    
        channels_dim = logits.ndim - 3
        n_classes = logits.shape[channels_dim]
        target_mask = torch.nn.Softmax(dim=channels_dim)(logits).argmax(dim=channels_dim).unsqueeze(dim=channels_dim).long()
        generated_mask = segmentation_logits_to_mask(logits)
        assert torch.all(target_mask == generated_mask)
        assert generated_mask.min() == 0
        assert generated_mask.max() == n_classes - 1
        assert target_mask.shape == generated_mask.shape
        assert target_mask.type() == generated_mask.type()


    @pytest.mark.parametrize(
        "logits",
        [
            # (np.random.rand(25, 25).astype(np.float32)),
            # (np.random.rand(1, 25, 25).astype(np.float32)),
            (np.random.rand(12, 1, 25, 25).astype(np.float32)),
        ],
    )
    def test_binary_segmentation_numpy(self, logits: FP32_TENSOR) -> None:

        target_mask = (logits >= 0.5).astype(np.long)
        generated_mask = segmentation_logits_to_mask(logits)
        # assert np.all(target_mask == generated_mask)
        assert generated_mask.min() == 0
        assert generated_mask.max() == 1
        assert target_mask.shape == generated_mask.shape
        assert target_mask.dtype == generated_mask.dtype
        
        