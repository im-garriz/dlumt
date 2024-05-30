import pytest
import numpy as np
import torch


class TestSegmentationLogitsToMask:
    
    def test_binary_segmentation_from_numpy(self) -> None:
        
        logits = np.random.rand()