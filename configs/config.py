import os
import torch

# Base Paths (Updated for local usage or specific environment if needed)
# NOTE: Update BASE_PATH to match your actual data location
BASE_PATH = '/workspace'

# Dataset 1 Paths
DS1_ORIGINAL_BENIGN = os.path.join(BASE_PATH, 'Dataset 1', 'original_data', 'benign_lesions')
DS1_ORIGINAL_MALIGNANT = os.path.join(BASE_PATH, 'Dataset 1', 'original_data', 'malignant_lesions')

# Dataset 2 Paths
DS2_TRAINING = os.path.join(BASE_PATH, 'Dataset 2 ', 'Training')
DS2_VALIDATION = os.path.join(BASE_PATH, 'Dataset 2 ', 'Validation')
DS2_TESTING = os.path.join(BASE_PATH, 'Dataset 2 ', 'Testing')

# Dataset configuration
DS2_CLASSES = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']
MALIGNANT_SUBTYPES = ['MC', 'OC', 'CaS']
NUM_SUBTYPES = len(DS2_CLASSES)

# Model configuration
IMG_SIZE = 224
BATCH_SIZE = 128  
NUM_WORKERS = 8  
# BACKBONE Options:
# 'resnet50'        - ResNet50 (Default)
# 'densenet121'     - DenseNet121
# 'convnext_tiny'   - ConvNeXt Tiny
# 'swin_t'          - Swin Transformer Tiny
# 'efficientnet_b0' - EfficientNet B0
# 'efficientnet_v2b2' - EfficientNet V2-B2
# 'efficientnet_v2b3' - EfficientNet V2-B3
# 'efficientnet_v2s'  - EfficientNet V2-S
BACKBONE = 'swin_t'
DROPOUT = 0.5
USE_PRETRAINED = False  
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Training configuration
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 200
SEED = 42

# Learning Rate Scheduler configuration
SCHEDULER_TYPE = 'cosine'  
SCHEDULER_PATIENCE = 5  
SCHEDULER_FACTOR = 0.5  
SCHEDULER_STEP_SIZE = 10  
SCHEDULER_GAMMA = 0.95  

# Early Stopping configuration
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15  
EARLY_STOPPING_MIN_DELTA = 1e-4  

# Paths for saving results
SAVE_DIR = os.path.join(BASE_PATH, 'results', BACKBONE)
BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
HISTORY_PLOT_PATH = os.path.join(SAVE_DIR, 'training_history.png')
CONFUSION_MATRIX_PATH = os.path.join(SAVE_DIR, 'confusion_matrices.png')

# Ensure save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
