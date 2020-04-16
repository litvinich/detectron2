from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from catalyst.utils import metrics
from torch.nn.modules.loss import _Loss

from detectron2.config import CfgNode


def onehot(indexes, N=None, ignore_index=None):
    """
    Creates a one-representation of indexes with N possible entries
    if N is not specified, it will suit the maximum index appearing.
    indexes is a long-tensor of indexes
    ignore_index will be zero in onehot representation
    """
    if N is None:
        N = indexes.max() + 1
    sz = list(indexes.size())
    output = indexes.new().byte().resize_(*sz, N).zero_()
    output.scatter_(-1, indexes.unsqueeze(-1), 1)
    if ignore_index is not None and ignore_index >= 0:
        output.masked_fill_(indexes.eq(ignore_index).unsqueeze(-1), 0)
    return output


def _is_long(x):
    if hasattr(x, "data"):
        x = x.data
    return isinstance(x, torch.LongTensor) or isinstance(x, torch.cuda.LongTensor)


def cross_entropy_label_smoothing(
    inputs,
    target,
    weight=None,
    ignore_index=-100,
    reduction="mean",
    smooth_eps=None,
    smooth_dist=None,
    from_logits=True,
):
    """cross entropy loss, with support for target distributions and label smoothing https://arxiv.org/abs/1512.00567"""
    smooth_eps = smooth_eps or 0

    # ordinary log-likelihood - use cross_entropy from nn
    if _is_long(target) and smooth_eps == 0:
        if from_logits:
            return F.cross_entropy(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)
        else:
            return F.nll_loss(inputs, target, weight, ignore_index=ignore_index, reduction=reduction)

    if from_logits:
        # log-softmax of inputs
        lsm = F.log_softmax(inputs, dim=-1)
    else:
        lsm = inputs

    masked_indices = None
    num_classes = inputs.size(-1)

    if _is_long(target) and ignore_index >= 0:
        masked_indices = target.eq(ignore_index)

    if smooth_eps > 0 and smooth_dist is not None:
        if _is_long(target):
            target = onehot(target, num_classes).type_as(inputs)
        if smooth_dist.dim() < target.dim():
            smooth_dist = smooth_dist.unsqueeze(0)
        target.lerp_(smooth_dist, smooth_eps)

    if weight is not None:
        lsm = lsm * weight.unsqueeze(0)

    if _is_long(target):
        eps_sum = smooth_eps / num_classes
        eps_nll = 1.0 - eps_sum - smooth_eps
        likelihood = lsm.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        loss = -(eps_nll * likelihood + eps_sum * lsm.sum(-1))
    else:
        loss = -(target * lsm).sum(-1)

    if masked_indices is not None:
        loss.masked_fill_(masked_indices, 0)

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        if masked_indices is None:
            loss = loss.mean()
        else:
            loss = loss.sum() / float(loss.size(0) - masked_indices.sum())
    return loss


class CrossEntropyLossLabelSmoothing(nn.CrossEntropyLoss):
    """CrossEntropyLoss - with ability to recieve distrbution as targets, and optional label smoothing"""

    def __init__(
        self, weight=None, ignore_index=-100, reduction="mean", smooth_eps=None, smooth_dist=None, from_logits=True
    ):
        super(CrossEntropyLossLabelSmoothing, self).__init__(
            weight=weight, ignore_index=ignore_index, reduction=reduction
        )
        self.smooth_eps = smooth_eps
        self.smooth_dist = smooth_dist
        self.from_logits = from_logits

    def forward(self, input, target, smooth_dist=None):
        if smooth_dist is None:
            smooth_dist = self.smooth_dist
        return cross_entropy_label_smoothing(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            smooth_eps=self.smooth_eps,
            smooth_dist=smooth_dist,
            from_logits=self.from_logits,
        )


class FocalLossBinary(_Loss):
    def __init__(
        self,
        ignore: int = None,
        reduced: bool = False,
        gamma: float = 2.0,
        alpha: float = 0.25,
        threshold: float = 0.5,
        reduction: str = "mean",
    ):
        """
        Compute focal loss for binary classification problem.
        """
        super().__init__()
        self.ignore = ignore

        if reduced:
            self.loss_fn = partial(metrics.reduced_focal_loss, gamma=gamma, threshold=threshold, reduction=reduction)
        else:
            self.loss_fn = partial(metrics.sigmoid_focal_loss, gamma=gamma, alpha=alpha, reduction=reduction)

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; ...]
            targets: [bs; ...]
        """
        targets = targets.view(-1)
        logits = logits.view(-1)

        if self.ignore is not None:
            # Filter predictions with ignore label from loss computation
            not_ignored = targets != self.ignore
            logits = logits[not_ignored]
            targets = targets[not_ignored]

        loss = self.loss_fn(logits, targets)
        return loss


class FocalLossMultiClass(FocalLossBinary):
    """
    Compute focal loss for multi-class problem.
    Ignores targets having -1 label
    """

    def forward(self, logits, targets):
        """
        Args:
            logits: [bs; num_classes; ...]
            targets: [bs; ...]
        """
        num_classes = logits.size(1)
        loss = 0
        targets = targets.view(-1)
        logits = logits.view(-1, num_classes)

        # Filter anchors with -1 label from loss computation
        if self.ignore is not None:
            not_ignored = targets != self.ignore

        for cls in range(num_classes):
            cls_label_target = (targets == (cls + 0)).long()
            cls_label_input = logits[..., cls]

            if self.ignore is not None:
                cls_label_target = cls_label_target[not_ignored]
                cls_label_input = cls_label_input[not_ignored]

            loss += self.loss_fn(cls_label_input, cls_label_target)
        return loss


def build_classification_loss(cfg: CfgNode) -> torch.nn.Module:
    name = cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS.NAME
    if name == "FocalLoss":
        return FocalLossMultiClass(
            gamma=cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS.GAMMA, alpha=cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS.ALPHA
        )
    elif name == "CrossEntropyLoss":
        return CrossEntropyLossLabelSmoothing(smooth_eps=cfg.MODEL.ROI_HEADS.CLASSIFICATION_LOSS.LABEL_SMOOTHING)
    else:
        raise ValueError("Unknown classification loss: {}".format(name))
