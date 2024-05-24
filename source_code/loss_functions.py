import torch
import torch.nn.functional as F
import numpy as np


class DiceLoss(torch.nn.Module):
    """
    Computes the Dice Loss, which is used for evaluating the performance of image segmentation models.
    """
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for DiceLoss.

        Parameters:
        inputs (torch.Tensor): Predicted outputs from the model.
        targets (torch.Tensor): Ground truth labels.
        smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        torch.Tensor: Dice loss value.
        """
        preds = F.softmax(inputs, dim=1)
        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds + targets, dim=(2, 3))

        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - torch.mean(dice_coeff)
        return dice_loss


class DiceLossWeighted(torch.nn.Module):
    """
    Computes the Weighted Dice Loss, a generalization of Dice Loss that applies weights to different classes.
    """
    def __init__(self, class_weight=np.ones(9)):
        super(DiceLossWeighted, self).__init__()
        self.class_weight = class_weight

    def forward(self, inputs, targets, smooth=1):
        """
        Forward pass for DiceLossWeighted.

        Parameters:
        inputs (torch.Tensor): Predicted outputs from the model.
        targets (torch.Tensor): Ground truth labels.
        smooth (float): Smoothing factor to prevent division by zero.

        Returns:
        torch.Tensor: Weighted Dice loss value.
        """
        preds = F.softmax(inputs, dim=1)
        intersection = torch.sum(preds * targets, dim=(2, 3))
        union = torch.sum(preds + targets, dim=(2, 3))

        class_weight_broadcast = self.class_weight[np.newaxis, :, np.newaxis, np.newaxis]

        intersection = intersection * class_weight_broadcast
        union = union * class_weight_broadcast

        dice_coeff = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1 - torch.mean(dice_coeff)
        return dice_loss


class FocalLoss(torch.nn.Module):
    """
    Computes the Focal Loss, which is used for addressing class imbalance by focusing on hard-to-classify examples.
    """
    def __init__(self, num_classes, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Forward pass for FocalLoss.

        Parameters:
        pred (torch.Tensor): Predicted outputs from the model.
        target (torch.Tensor): Ground truth labels.

        Returns:
        torch.Tensor: Focal loss value.
        """
        pred_probs = F.softmax(pred, dim=1)
        cross_entropy_loss = F.binary_cross_entropy(pred_probs, target, reduction='none')

        if self.alpha is None:
            alpha = torch.ones(self.num_classes)
        else:
            alpha = torch.tensor(self.alpha)

        alpha = alpha.to(pred_probs.device)
        alpha_broadcast = alpha.view(1, self.num_classes, 1, 1)
        alpha_factor = alpha_broadcast * target + (1 - alpha_broadcast) * (1 - target)
        modulating_factor = (1 - pred_probs) ** self.gamma

        focal_loss = alpha_factor * modulating_factor * cross_entropy_loss
        focal_loss = torch.mean(focal_loss)

        return focal_loss
