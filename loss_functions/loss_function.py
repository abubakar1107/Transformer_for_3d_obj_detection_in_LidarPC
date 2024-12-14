import torch
import torch.nn as nn

class CombinedLoss(nn.Module):
    def __init__(self, alpha=2.5, beta=2):
        super(CombinedLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_boxes, true_boxes, pred_classes, true_classes):

    
        # Define class weights (example: adjust based on dataset distribution)
        class_weights = torch.tensor([1.0, 2.0, 1.5], device=true_classes.device)  # Adjust weights per class
        classification_loss = nn.CrossEntropyLoss(weight=class_weights)(pred_classes, true_classes)

        # print(f"pred_boxes shape: {pred_boxes.shape}")  # Should be [batch_size, 7]
        # print(f"true_boxes shape: {true_boxes.shape}")  # Should be [batch_size, 7]
        # Smooth L1 and IoU Loss
        reg_loss = self.smooth_l1(pred_boxes, true_boxes)
        # print(f"Inside CombinedLoss - pred_boxes shape: {pred_boxes.shape}")

        iou_loss = 1 - self.calculate_iou(pred_boxes, true_boxes)

        # Combined Loss
        total_loss = self.alpha * reg_loss + self.beta * iou_loss + classification_loss
        return total_loss


    def calculate_iou(self, pred_boxes, true_boxes):
        pred_boxes = pred_boxes[:, :6]  # Explicitly include only 6 elements
        true_boxes = true_boxes[:, :6]

        # Compute intersection
        inter_min = torch.max(pred_boxes[:, :3], true_boxes[:, :3])  # Intersection min corner
        inter_max = torch.min(pred_boxes[:, 3:6], true_boxes[:, 3:6])  # Intersection max corner
        # print(f"inter_min shape: {inter_min.shape}")
        # print(f"inter_max shape: {inter_max.shape}")

        inter_dims = torch.clamp(inter_max - inter_min, min=0)  # Dimensions of the intersection
        # print(f"inter_dims shape: {inter_dims.shape}")

        intersection = inter_dims.prod(dim=1)  # Volume of the intersection
        # print(f"intersection shape: {intersection.shape}")

        pred_vol = (pred_boxes[:, 3:] - pred_boxes[:, :3]).prod(dim=1)  # Volume of predicted box
        true_vol = (true_boxes[:, 3:] - true_boxes[:, :3]).prod(dim=1)  # Volume of ground truth box
        union = pred_vol + true_vol - intersection  # Union volume

        iou = intersection / (union + 1e-6)
        return iou



