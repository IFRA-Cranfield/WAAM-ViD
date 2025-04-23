import torch
import torch.nn as nn
import torch.nn.functional as F
    
def dice_metrics(inputs, targets, smooth=1e-8):
    inputs = (inputs >= 0.5)
    targets = (targets >= 0.5)
 
    inputs_sum = inputs.sum()
    targets_sum = targets.sum()
 
    if inputs_sum == 0 and targets_sum == 0:
        return torch.tensor(1.0)  # perfect match when both are empty
 
    intersection = (inputs & targets).sum()
    dice = (2. * intersection + smooth) / (inputs_sum + targets_sum + smooth)
 
    return dice


def precision_metrics(inputs, targets, smooth=1e-8):
    inputs = (inputs >= 0.5)
    targets = (targets >= 0.5)
 
    inputs_sum = inputs.sum()
    if inputs_sum == 0:
        return torch.tensor(1.0 if targets.sum() == 0 else 0.0)
 
    intersection = (inputs & targets).sum()
    precision = (intersection + smooth) / (inputs_sum + smooth)
 
    return precision
    

def recall_metrics(inputs, targets, smooth=1e-8):
    inputs = (inputs >= 0.5)
    targets = (targets >= 0.5)
 
    targets_sum = targets.sum()
    if targets_sum == 0:
        return torch.tensor(1.0 if inputs.sum() == 0 else 0.0)
 
    intersection = (inputs & targets).sum()
    recall = (intersection + smooth) / (targets_sum + smooth)
 
    return recall 


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1e-8):
        # fix for inputs and targets to be in the same shape
        if isinstance(inputs, dict) and "out" in inputs:
            inputs = inputs["out"]

        if inputs.shape != targets.shape:
            inputs = F.interpolate(inputs, size=targets.shape[2:], mode="bilinear", align_corners=False)

        # ensure inputs and targets are in the range [0, 1]
        inputs = torch.sigmoid(inputs)
        inputs = torch.clamp(inputs, 0, 1)
        targets = torch.clamp(targets, 0, 1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # calculate Dice coefficient
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        dice = (2. * intersection + smooth) / (union + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super(DiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()  # BCE Loss (Logits version)

    def forward(self, inputs, targets, smooth=1e-8, target_class_index=1):
        dice = DiceLoss()(inputs, targets, smooth)
        bce = self.bce(inputs, targets.float())
        # bce = self.bce(inputs[:, target_class_index, :, :], targets.float())  # BCE Loss
        return self.weight_dice * dice + self.weight_bce * bce
