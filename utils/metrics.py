"""
This file will contain the metrics of the framework
COMPLETE_BOX_IOU_LOSS can be used as a loss function from 
https://pytorch.org/vision/stable/generated/torchvision.ops.complete_box_iou_loss.html
"""
import numpy as np
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchmetrics.classification import MulticlassAveragePrecision
from torchmetrics.classification import MulticlassF1Score
import torch
from icecream import ic

class AverageMeter:
    """
    Class to be an average meter for any average metric like loss, accuracy, etc..
    """

    def __init__(self):
        self.momentum = 0.95 # 0 is better than 0.95
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        if not self.momentum:
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count
        
        elif self.momentum:
            self.val = val
            if self.count == 0:
                self.avg = self.val
            else:
                self.avg = self.avg*self.momentum + (1-self.momentum)* val
            self.count += n
        
    # @property
    # def val(self):
    #     return self.avg


class TrajectoryEvaluator:
    def __init__(self, cfg, predictions, targets):
        self.predictions = predictions  # Predicted trajectories (torch tensor) (batch_size, num_agents, K, seq_len, 2)
        self.targets = torch.unsqueeze(targets, dim = -3)          # Ground truth trajectories (torch tensor) (batch_size, num_agents, 1, seq_len, 2)
        self.cfg = cfg

    def calculate_fde(self):
        if self.cfg.MODEL.USE_VAE:
            final_pred_positions = self.predictions[:, :, :, -1, :] #(batch_size, num_agents, K, 2)
            final_target_positions = self.targets[:, :, :, -1, :] # (batch_size, num_agents, 1, 2)
        else:
            final_pred_positions = self.predictions[:, :, -1, :]
            final_target_positions = self.targets[:, :, -1, :]
        
        fde_values = torch.norm(final_pred_positions - final_target_positions, dim=-1) # (batch_size, num_agents, K)
        min_fde_values, _ = torch.min(fde_values, dim = -1) # (batch_size, num_agents)
        return min_fde_values 
    
    def calculate_ade(self):
        distances = torch.norm(self.predictions - self.targets, dim=-1) # (batch_size, num_agents, K, seq_len)
        ade_values = torch.mean(distances, dim=-1) # (batch_size, num_agents, K)
        min_ade_values, _ = torch.min(ade_values, dim = -1) # (batch_size, num_agents)
        return min_ade_values


class BoundingBoxEvaluator:
    def __init__(self, predictions, targets):
        """
        predictions and targets are tensors of shape (batch_size, num_agents, seq_len, 4)
        where each 4D vector represents the bounding box in the form [x_min, y_min, x_max, y_max].
        """
        self.predictions = predictions
        self.targets = targets

    def calculate_center_fde(self):
        final_pred_centers = (self.predictions[:, :, -1, :2] + self.predictions[:, :, -1, 2:]) / 2
        final_target_centers = (self.targets[:, :, -1, :2] + self.targets[:, :, -1, 2:]) / 2
        fde_values = torch.norm(final_pred_centers - final_target_centers, dim=2)
        return fde_values

    def calculate_center_ade(self):
        pred_centers = (self.predictions[..., :2] + self.predictions[..., 2:]) / 2
        target_centers = (self.targets[..., :2] + self.targets[..., 2:]) / 2
        distances = torch.norm(pred_centers - target_centers, dim=3)
        ade_values = torch.mean(distances, dim=2)
        return ade_values

    def calculate_arb(self):
        distances = torch.norm(self.predictions - self.targets, dim=3)
        arb_values = torch.sqrt(torch.mean(distances ** 2, dim=2))  # RMSE over the sequence
        return arb_values

    def calculate_frb(self):
        final_pred_boxes = self.predictions[:, :, -1, :]
        final_target_boxes = self.targets[:, :, -1, :]
        frb_values = torch.sqrt(torch.mean((final_pred_boxes - final_target_boxes) ** 2, dim=2))
        return frb_values

    def calculate_mse(self):
        mse_values = torch.mean((self.predictions - self.targets) ** 2, dim=(2, 3))  # MSE over the sequence
        return mse_values
