import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    """
    Combined loss for Multi-Task Learning.
    Loss 1 (Binary): CrossEntropyLoss for all samples
    Loss 2 (Subtype): CrossEntropyLoss with ignore_index=-1 (Masking Trick)
    """
    def __init__(self, weight_binary=1.0, weight_subtype=1.0):
        super(MultiTaskLoss, self).__init__()
        self.weight_binary = weight_binary
        self.weight_subtype = weight_subtype
        
        self.criterion_binary = nn.CrossEntropyLoss()
        self.criterion_subtype = nn.CrossEntropyLoss(ignore_index=-1)
    
    def forward(self, pred_binary, pred_subtype, target_binary, target_subtype):
        # Binary loss (all samples contribute)
        loss_binary = self.criterion_binary(pred_binary, target_binary)
        
        # Subtype loss (only DS2 samples contribute via ignore_index)
        loss_subtype = self.criterion_subtype(pred_subtype, target_subtype)
        
        # Handle NaN if batch has only DS1 samples
        if torch.isnan(loss_subtype):
            loss_subtype = torch.tensor(0.0, device=pred_binary.device)
        
        total_loss = (self.weight_binary * loss_binary) + (self.weight_subtype * loss_subtype)
        
        return total_loss, loss_binary, loss_subtype
