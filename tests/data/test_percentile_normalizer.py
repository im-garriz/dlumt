import numpy as np
import pytest
import torch

from dlumt.data.normalization import PercentilesNormalizer
from dlumt.typing import *

# pip install -e .


def get_random_tensor(type: str, min: int, max: int) -> FP32_TENSOR:

    if type == "numpy":
        x = np.random.rand(3, 8, 8).astype(np.float32)
    else:
        x = torch.rand(3, 8, 8).type(torch.float32)

    # we ensure a 0 and a 1 on each channel
    for ch in range(3):
        x[ch, 0, 0] = 0.0
        x[ch, 0, 1] = 1.0

    return min + x * (max - min)


class TestPercentilesNormalizer:

    @pytest.fixture
    def min_percentile(self) -> np.array:
        return np.array([5.0, 4.0, 8.0], dtype=np.float32)

    @pytest.fixture
    def max_percentile(self) -> np.array:
        return np.array([245.0, 224.0, 234.0], dtype=np.float32)

    @pytest.fixture
    def percentiles_normalizer(
        self, min_percentile: np.array, max_percentile: np.array
    ) -> PercentilesNormalizer:
        return PercentilesNormalizer(min_percentile, max_percentile)

    def test_check_input_raises_excp(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:
        """
        it has to throw exceptcion because we are passing it a list but
        it only supports np.array or torch.tensor
        """
        x = [1, 2, 4]  # List
        with pytest.raises(TypeError):
            _ = percentiles_normalizer.check_input(x)

    @pytest.mark.parametrize(
        "x,is_torch_tensor,device",
        [
            (255 * np.random.rand(3, 8, 8).astype(np.float32), False, None),
            (torch.rand(3, 8, 8), True, torch.device("cpu")),
            (torch.rand(3, 8, 8).to("cuda:0"), True, torch.device("cuda:0")),
        ],
    )
    def test_is_torch_and_device_are_correctly_preserved(
        self,
        x: FP32_TENSOR,
        is_torch_tensor: bool,
        device: torch.device,
        percentiles_normalizer: PercentilesNormalizer,
    ) -> None:

        is_torch_tensor_from_test, device_from_test = (
            percentiles_normalizer.check_input(x)
        )

        assert is_torch_tensor_from_test == is_torch_tensor
        assert device_from_test == device

    def test_normalize_is_ranged_0_1(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x = get_random_tensor("numpy", 0, 255)
        x_norm = percentiles_normalizer.normalize(x)

        assert 0 <= x_norm.max() <= 1
        assert 0 <= x_norm.min() <= 1

    def test_denormalize_is_ranged_min_max(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:
        """
        As get_random_tensor always has in all channels both the minimum and the
        maximum, when denormalising the maxima and minima are
        the maximum and minimum percentiles.
        """
        x = get_random_tensor("numpy", 0, 1)
        x_dnorm = percentiles_normalizer.denormalize(x)

        assert x_dnorm.max() == 245
        assert x_dnorm.min() == 4

    @pytest.mark.parametrize(
        "x",
        [
            (get_random_tensor("torch", 0, 255)),
            (get_random_tensor("torch", 0, 255).to("cuda:0")),
            (get_random_tensor("torch", 0, 255).to("cpu")),
        ],
    )
    def test_normalize_works_with_torch(
        self, percentiles_normalizer: PercentilesNormalizer, x: FP32_TENSOR
    ) -> None:

        x_norm = percentiles_normalizer.normalize(x)

        assert 0 <= x_norm.max() <= 1
        assert 0 <= x_norm.min() <= 1
        assert isinstance(x_norm, type(x))
        assert x_norm.device == x.device
        assert x.shape == x_norm.shape

    def test_normalize_and_denormalize_gives_same(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x = get_random_tensor("torch", 10, 220)
        x_norm = percentiles_normalizer.normalize(x)
        x_dnorm = percentiles_normalizer.denormalize(x_norm)

        assert torch.allclose(x, x_dnorm)

    def test_normalize_and_denormalize_dont_give_same_when_out_of_percs(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:
        """
        Here it does not have to be the same because the value 255
        will be clipped to normalise, as it is higher than the percentiles defined above.
        """
        x = get_random_tensor("torch", 10, 255)
        x_norm = percentiles_normalizer.normalize(x)
        x_dnorm = percentiles_normalizer.denormalize(x_norm)

        assert x_dnorm.max() < x.max()

    def test_n_channels(self, percentiles_normalizer: PercentilesNormalizer) -> None:
        assert percentiles_normalizer.n_channels == 3

    def test_3_dim_and_4_dim_get_same_normalization_numpy(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x_1 = get_random_tensor("numpy", 10, 255)
        x_2 = get_random_tensor("numpy", 10, 255)

        x_merge = np.concatenate([x_1[None, ...], x_2[None, ...]], axis=0)

        x_1_normalized = percentiles_normalizer.normalize(x_1)
        x_2_normalized = percentiles_normalizer.normalize(x_2)
        x_merge_normalized = percentiles_normalizer.normalize(x_merge)

        x_12_normalized = np.concatenate(
            [x_1_normalized[None, ...], x_2_normalized[None, ...]], axis=0
        )

        assert np.allclose(x_12_normalized, x_merge_normalized)

    def test_3_dim_and_4_dim_get_same_denormalization_numpy(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x_1 = get_random_tensor("numpy", 0, 1)
        x_2 = get_random_tensor("numpy", 0, 1)

        x_merge = np.concatenate([x_1[None, ...], x_2[None, ...]], axis=0)

        x_1_denormalized = percentiles_normalizer.denormalize(x_1)
        x_2_denormalized = percentiles_normalizer.denormalize(x_2)
        x_merge_denormalized = percentiles_normalizer.denormalize(x_merge)

        x_12_denormalized = np.concatenate(
            [x_1_denormalized[None, ...], x_2_denormalized[None, ...]], axis=0
        )

        assert np.allclose(x_12_denormalized, x_merge_denormalized)

    def test_3_dim_and_4_dim_get_same_normalization_torch(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x_1 = get_random_tensor("torch", 10, 255)
        x_2 = get_random_tensor("torch", 10, 255)

        x_merge = torch.cat([x_1[None, ...], x_2[None, ...]], dim=0)

        x_1_normalized = percentiles_normalizer.normalize(x_1)
        x_2_normalized = percentiles_normalizer.normalize(x_2)
        x_merge_normalized = percentiles_normalizer.normalize(x_merge)

        x_12_normalized = torch.cat(
            [x_1_normalized[None, ...], x_2_normalized[None, ...]], dim=0
        )

        assert torch.allclose(x_12_normalized, x_merge_normalized)

    def test_3_dim_and_4_dim_get_same_denormalization_torch(
        self, percentiles_normalizer: PercentilesNormalizer
    ) -> None:

        x_1 = get_random_tensor("torch", 0, 1)
        x_2 = get_random_tensor("torch", 0, 1)

        x_merge = torch.cat([x_1[None, ...], x_2[None, ...]], dim=0)

        x_1_denormalized = percentiles_normalizer.denormalize(x_1)
        x_2_denormalized = percentiles_normalizer.denormalize(x_2)
        x_merge_denormalized = percentiles_normalizer.denormalize(x_merge)

        x_12_denormalized = torch.cat(
            [x_1_denormalized[None, ...], x_2_denormalized[None, ...]], dim=0
        )

        assert torch.allclose(x_12_denormalized, x_merge_denormalized)
