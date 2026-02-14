import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from PIL import Image
from configs.config import DS2_CLASSES
from data.transforms import val_transform

def evaluate_model(model, test_loader, device):
    model.eval()
    results = {
        'preds_binary': [], 'targets_binary': [],
        'preds_subtype': [], 'targets_subtype': []
    }
    
    with torch.no_grad():
        for images, targets_b, targets_s in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            pred_b, pred_s = model(images)
            
            results['preds_binary'].extend(torch.argmax(pred_b, dim=1).cpu().numpy())
            results['targets_binary'].extend(targets_b.numpy())
            
            mask = targets_s != -1
            if mask.sum() > 0:
                results['preds_subtype'].extend(torch.argmax(pred_s[mask], dim=1).cpu().numpy())
                results['targets_subtype'].extend(targets_s[mask].numpy())
                
    return {k: np.array(v) for k, v in results.items()}

def plot_confusion_matrix(y_true, y_pred, classes, title, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

def predict_single_image(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    tensor = val_transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_b, pred_s = model(tensor)
        prob_b = torch.softmax(pred_b, dim=1)[0]
        prob_s = torch.softmax(pred_s, dim=1)[0]
        
        idx_b = torch.argmax(prob_b).item()
        idx_s = torch.argmax(prob_s).item()
    
    return {
        'binary': ('Malignant' if idx_b == 1 else 'Benign', prob_b[idx_b].item()),
        'subtype': (DS2_CLASSES[idx_s], prob_s[idx_s].item())
    }
