import pytest
import numpy as np
import torch

from dlumt.segmentation.metrics import calc_metrics_multiclass, calc_metrics_binary 
from dlumt.typing import *

class TestSegmentationMetrics:
    
    @pytest.mark.parametrize(
        "y_hat,target",
        [
            (torch.randint(0, 1, (1, 50, 50)).type(torch.uint8), torch.randint(0, 1, (1, 50, 50)).type(torch.uint8)),
            (torch.randint(0, 1, (1, 50, 50)).type(torch.int16), torch.randint(0, 1, (1, 50, 50)).type(torch.int16)),
            (torch.randint(0, 1, (1, 50, 50)).type(torch.int32), torch.randint(0, 1, (1, 50, 50)).type(torch.int32)),
        ],
    )
    def test_incorrect_data_types_raises_error(self, y_hat: INT64_TENSOR, target: INT64_TENSOR) -> None:
        
        with pytest.raises(TypeError):
            _ = calc_metrics_multiclass(y_hat, target)
            
        with pytest.raises(TypeError):
            _ = calc_metrics_binary(y_hat, target)
            

    @pytest.mark.parametrize(
        "y_hat,target",
        [
            (torch.randint(0, 1, (1, 1, 50, 50)).type(torch.int64), torch.randint(0, 1, (1, 50, 50)).type(torch.int64)),
            (torch.randint(0, 1, (1, 50, 48)).type(torch.int64), torch.randint(0, 1, (1, 50, 50)).type(torch.int64)),
        ],
    )   
    def test_different_shape_raises_error(self, y_hat: INT64_TENSOR, target: INT64_TENSOR) -> None:
        
        with pytest.raises(AssertionError):
            _ = calc_metrics_multiclass(y_hat, target, n_classes=5, classes_dict={})
            
        with pytest.raises(AssertionError):
            _ = calc_metrics_binary(y_hat, target)
            
            
    @pytest.mark.parametrize(
        "y_hat,target",
        [
            (torch.randint(0, 5, (50, 50)).type(torch.int64), torch.randint(0, 5, (50, 50)).type(torch.int64)),
            (torch.randint(0, 6, (1, 1, 50, 50)).type(torch.int64), torch.randint(0, 6, (1, 50, 50)).type(torch.int64)),
        ],
    )   
    def test_incorrect_shape_raises_error_multiclass(self, y_hat: INT64_TENSOR, target: INT64_TENSOR) -> None:
        
        with pytest.raises(AssertionError):
            _ = calc_metrics_multiclass(y_hat, target, n_classes=5, classes_dict={})
            
            
    @pytest.mark.parametrize(
        "y_hat,target",
        [
            (torch.randint(0, 1, (1, 50, 50)).type(torch.int64), torch.randint(0, 1, (1, 50, 50)).type(torch.int64)),
            (torch.randint(0, 1, (50, 50)).type(torch.int64), torch.randint(0, 1, (50, 50)).type(torch.int64)),
        ],
    )   
    def test_incorrect_shape_raises_error_binary(self, y_hat: INT64_TENSOR, target: INT64_TENSOR) -> None:
        
        with pytest.raises(AssertionError):
            _ = calc_metrics_binary(y_hat, target)
            
            
    @pytest.mark.parametrize(
        "y_hat,target",
        [
            (torch.randint(0, 1, (4, 2, 50, 50)).type(torch.int64), torch.randint(0, 1, (4, 2, 50, 50)).type(torch.int64)),
        ],
    )   
    def test_incorrect_channels_raises_error_binary(self, y_hat: INT64_TENSOR, target: INT64_TENSOR) -> None:
        
        with pytest.raises(AssertionError):
            _ = calc_metrics_binary(y_hat, target)
