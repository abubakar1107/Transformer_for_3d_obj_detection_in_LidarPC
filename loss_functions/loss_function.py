import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(CombinedLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_boxes, true_boxes, pred_classes, true_classes):
        # Smooth L1 for bounding box regression
        reg_loss = self.smooth_l1(pred_boxes, true_boxes)
        
        # IoU loss
        iou_loss = 1 - self.calculate_iou(pred_boxes, true_boxes)
        
        # Cross-Entropy Loss for classification
        classification_loss = nn.CrossEntropyLoss()(pred_classes, true_classes)
        
        # Total combined loss
        total_loss = self.alpha * reg_loss + self.beta * iou_loss + classification_loss
        return total_loss

    def calculate_iou(self, pred_boxes, true_boxes):
        # Calculate intersection
        inter_min = torch.max(pred_boxes[:, :3], true_boxes[:, :3])
        inter_max = torch.min(pred_boxes[:, 3:], true_boxes[:, 3:])
        inter_dims = torch.clamp(inter_max - inter_min, min=0)
        intersection = inter_dims[:, 0] * inter_dims[:, 1] * inter_dims[:, 2]
        
        # Calculate union
        pred_vol = (pred_boxes[:, 3] - pred_boxes[:, 0]) * (pred_boxes[:, 4] - pred_boxes[:, 1]) * (pred_boxes[:, 5] - pred_boxes[:, 2])
        true_vol = (true_boxes[:, 3] - true_boxes[:, 0]) * (true_boxes[:, 4] - true_boxes[:, 1]) * (true_boxes[:, 5] - true_boxes[:, 2])
        union = pred_vol + true_vol - intersection

        iou = intersection / (union + 1e-6)
        return iou
