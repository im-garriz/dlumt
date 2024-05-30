from typing import Optional, Mapping, Union
import torch
import torch.nn as nn
import ttach as tta


class SegmentationTTAWrapper(nn.Module):
    """
    A wrapper for segmentation models to apply Test Time Augmentation (TTA).

    This wrapper augments the input image using a set of transformations,
    applies the model to each augmented image, de-augments the output masks,
    and then merges the results using a specified merge mode.

    Args:
        model (nn.Module): The segmentation model to be wrapped.
        transforms (tta.base.Compose): A set of transformations to apply for TTA.
        merge_mode (str): The method to merge the predictions from augmented images.
                          Options are 'mean', 'max', 'min', etc. Default is 'mean'.
        output_mask_key (Optional[str]): If the model's output is a dictionary, specify
                                         the key for the output mask. Default is None.
        temperature (float): Temperature scaling factor for the output mask. Default is 0.5.
        after_model_function (Optional[nn.Module]): An optional function to apply to the
                                                    model's output before de-augmentation.
                                                    Default is None.
    """

    def __init__(
        self,
        model: nn.Module,
        transforms: tta.base.Compose,
        merge_mode: str = "mean",
        output_mask_key: Optional[str] = None,
        temperature: float = 0.5,
        after_model_function: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.model = model
        self.transforms = transforms
        self.merge_mode = merge_mode
        self.output_key = output_mask_key
        self.temperature = temperature
        self.after_model_function = after_model_function

    def forward(
        self, image: torch.Tensor, *args
    ) -> Union[torch.Tensor, Mapping[str, torch.Tensor]]:
        """
        Forward pass for the SegmentationTTAWrapper.

        This method augments the input image, applies the model to each augmented image,
        de-augments the output masks, and merges the results.

        Args:
            image (torch.Tensor): The input image tensor.
            *args: Additional arguments to pass to the model's forward method.

        Returns:
            Union[torch.Tensor, Mapping[str, torch.Tensor]]: The merged output mask,
                                                             optionally wrapped in a dictionary.
        """
        merger = tta.base.Merger(type=self.merge_mode, n=len(self.transforms))

        for transformer in self.transforms:
            augmented_image = transformer.augment_image(image)
            augmented_output = self.model(augmented_image, *args)
            if self.output_key is not None:
                augmented_output = augmented_output[self.output_key]
            if self.after_model_function is not None:
                augmented_output = self.after_model_function(augmented_output)
            deaugmented_output = transformer.deaugment_mask(
                augmented_output**self.temperature
            )
            merger.append(deaugmented_output)

        result = merger.result
        if self.output_key is not None:
            result = {self.output_key: result}

        return result
