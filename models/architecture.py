import torch
import torch.nn as nn
import timm
from configs.config import NUM_SUBTYPES, BACKBONE, DROPOUT, USE_PRETRAINED

# Map user-friendly names to timm model names
BACKBONE_MAP = {
    'resnet50': 'resnet50',
    'densenet121': 'densenet121',
    'convnext_tiny': 'convnext_tiny',
    'swin_t': 'swin_tiny_patch4_window7_224',
    'efficientnet_b0': 'efficientnet_b0',
    'efficientnet_v2b2': 'tf_efficientnetv2_b2',
    'efficientnet_v2b3': 'tf_efficientnetv2_b3',
    'efficientnet_v2s': 'tf_efficientnetv2_s',
}

class MultiTaskOralClassifier(nn.Module):
    """
    Dual-Head Multi-Task Model for Oral Pathology Classification.
    Shared Backbone with Two Independent Parallel Heads.
    Uses 'timm' for flexible backbone selection.
    """
    def __init__(self, backbone=None, num_subtypes=None, dropout=None, pretrained=None):
        super(MultiTaskOralClassifier, self).__init__()
        
        # Resolve config with arguments or defaults
        self.backbone_name = backbone if backbone else BACKBONE
        self.num_subtypes = num_subtypes if num_subtypes is not None else NUM_SUBTYPES
        self.dropout_val = dropout if dropout is not None else DROPOUT
        self.use_pretrained = pretrained if pretrained is not None else USE_PRETRAINED
        
        # Determine timm model name
        if self.backbone_name not in BACKBONE_MAP:
            raise ValueError(f"Unsupported backbone: {self.backbone_name}. Supported: {list(BACKBONE_MAP.keys())}")
            
        timm_model_name = BACKBONE_MAP[self.backbone_name]
        print(f"Initializing {self.backbone_name} ({timm_model_name}) with pretrained={self.use_pretrained}")
        
        # Initialize backbone using timm
        self.backbone = timm.create_model(
            timm_model_name, 
            pretrained=self.use_pretrained, 
            num_classes=0
        )
        
        # Automatically get the number of output features from the backbone
        num_features = self.backbone.num_features
        
        self.dropout_layer = nn.Dropout(p=self.dropout_val)
        
        # Head 1: Binary Classification (Malignant vs Benign)
        self.head_binary = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_val),
            nn.Linear(512, 2)
        )
        
        # Head 2: Subtype Classification
        self.head_subtype = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_val),
            nn.Linear(512, self.num_subtypes)
        )
        
        print(f"Model initialized with {self.backbone_name} backbone (features={num_features})")
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout_layer(features)
        
        out_binary = self.head_binary(features)
        out_subtype = self.head_subtype(features)
        
        return out_binary, out_subtype
    
    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        print("Backbone frozen.")
    
    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
        print("Backbone unfrozen.")
