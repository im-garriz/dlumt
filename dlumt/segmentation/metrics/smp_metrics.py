from typing import Dict, Union, List, Optional
import segmentation_models_pytorch as smp
import torch

from dlumt.typing import *


def calc_metrics_multiclass(
    y_hat: INT64_TENSOR,
    y: INT64_TENSOR,
    n_classes: int,
    classes_dict: Dict[str, str],
    ignore_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    check_tensor_is_INT64_torch_tensor(y)
    check_tensor_is_INT64_torch_tensor(y_hat)

    assert (
        y_hat.shape == y.shape
    ), f"Prediction and target must have the same shape, got {y_hat.shape} and {y.shape}"
    assert (
        y.ndim == 3
    ), f"The shape of the input tensors must be of the type (C, H, W), got {y.ndim} dimensional"

    metrics = {}

    tp, fp, fn, tn = smp.metrics.get_stats(
        y_hat, y, mode="multiclass", num_classes=n_classes, ignore_index=ignore_index
    )

    metrics = get_metrics_macro(tp, fp, fn, tn)

    for class_idx in range(n_classes):
        class_ = classes_dict[f"{class_idx}"]
        f1_ = smp.metrics.f1_score(
            tp[:, class_idx],
            fp[:, class_idx],
            fn[:, class_idx],
            tn[:, class_idx],
            reduction="micro",
        )
        if not f1_.isnan():
            metrics.update({f"micro_f1_{class_}": f1_})
        else:
            metrics.update({f"micro_f1_{class_}": 1.0})

    return metrics


def calc_metrics_single_class(
    y_hat: INT64_TENSOR, y: INT64_TENSOR, ignore_index: Optional[int] = None,
) -> Dict[str, torch.Tensor]:

    check_tensor_is_INT64_torch_tensor(y)
    check_tensor_is_INT64_torch_tensor(y_hat)

    assert (
        y_hat.shape == y.shape
    ), f"Prediction and target must have the same shape, got {y_hat.shape} and {y.shape}"
    assert (
        y.ndim == 4
    ), f"The shape of the input tensors must be of the type (N, 1, H, W), got {y.ndim} dimensional"
    assert (
        y.shape[1] == 1
    ), f"The shape of the input tensors must be of the type (N, 1, H, W), got {y.shape}"

    tp, fp, fn, tn = smp.metrics.get_stats(y_hat, y, mode="binary", ignore_index=ignore_index)

    metrics = get_metrics_macro(tp, fp, fn, tn)

    return metrics


def get_metrics_macro(
    tp: Tuple[torch.LongTensor],
    fp: Tuple[torch.LongTensor],
    fn: Tuple[torch.LongTensor],
    tn: Tuple[torch.LongTensor],
    from_binary: Optional[bool] = False,
) -> Dict[str, torch.Tensor]:

    metrics = {}

    metrics.update(
        {"macro_f1": smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro")}
    )
    metrics.update(
        {"macro_iou": smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro")}
    )

    metrics.update(
        {
            "macro_avg_precision": smp.metrics.precision(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {"macro_recall": smp.metrics.recall(tp, fp, fn, tn, reduction="macro")}
    )
    metrics.update(
        {
            "macro_sensitivity": smp.metrics.sensitivity(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_specificity": smp.metrics.specificity(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )

    metrics.update(
        {
            "macro_balanced_accuracy": smp.metrics.balanced_accuracy(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_positive_predictive_value": smp.metrics.positive_predictive_value(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_averaged_negative_predictive_value": smp.metrics.negative_predictive_value(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )

    metrics.update(
        {
            "macro_false_negative_rate": smp.metrics.false_negative_rate(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_false_positive_rate": smp.metrics.false_positive_rate(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_false_discovery_rate": smp.metrics.false_discovery_rate(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_false_omission_rate": smp.metrics.false_omission_rate(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )

    metrics.update(
        {
            "macro_positive_likelihood_ratio": smp.metrics.positive_likelihood_ratio(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )
    metrics.update(
        {
            "macro_negative_likelihood_ratio": smp.metrics.negative_likelihood_ratio(
                tp, fp, fn, tn, reduction="macro"
            )
        }
    )

    if from_binary:
        return {
            metric.replace("macro", "binary"): value
            for metric, value in metrics.items()
        }

    return metrics
