import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss, running_loss_b, running_loss_s = 0.0, 0.0, 0.0
    all_preds_b, all_targets_b = [], []
    all_preds_s, all_targets_s = [], []
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, targets_b, targets_s in pbar:
        images, targets_b, targets_s = images.to(device), targets_b.to(device), targets_s.to(device)
        
        optimizer.zero_grad()
        pred_b, pred_s = model(images)
        loss, loss_b, loss_s = criterion(pred_b, pred_s, targets_b, targets_s)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        running_loss_b += loss_b.item()
        running_loss_s += loss_s.item() if not torch.isnan(loss_s) else 0
        
        preds_b = torch.argmax(pred_b, dim=1)
        all_preds_b.extend(preds_b.cpu().numpy())
        all_targets_b.extend(targets_b.cpu().numpy())
        
        mask = targets_s != -1
        if mask.sum() > 0:
            preds_s = torch.argmax(pred_s[mask], dim=1)
            all_preds_s.extend(preds_s.cpu().numpy())
            all_targets_s.extend(targets_s[mask].cpu().numpy())
            
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    avg_loss = running_loss / len(train_loader)
    avg_loss_b = running_loss_b / len(train_loader)
    avg_loss_s = running_loss_s / len(train_loader)
    acc_b = accuracy_score(all_targets_b, all_preds_b)
    acc_s = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    
    return avg_loss, avg_loss_b, avg_loss_s, acc_b, acc_s

def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds_b, all_targets_b = [], []
    all_preds_s, all_targets_s = [], []
    
    with torch.no_grad():
        for images, targets_b, targets_s in tqdm(val_loader, desc="Validating", leave=False):
            images, targets_b, targets_s = images.to(device), targets_b.to(device), targets_s.to(device)
            pred_b, pred_s = model(images)
            loss, _, _ = criterion(pred_b, pred_s, targets_b, targets_s)
            running_loss += loss.item()
            
            preds_b = torch.argmax(pred_b, dim=1)
            all_preds_b.extend(preds_b.cpu().numpy())
            all_targets_b.extend(targets_b.cpu().numpy())
            
            mask = targets_s != -1
            if mask.sum() > 0:
                preds_s = torch.argmax(pred_s[mask], dim=1)
                all_preds_s.extend(preds_s.cpu().numpy())
                all_targets_s.extend(targets_s[mask].cpu().numpy())
    
    avg_loss = running_loss / len(val_loader)
    acc_b = accuracy_score(all_targets_b, all_preds_b)
    acc_s = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    
    return avg_loss, acc_b, acc_s
