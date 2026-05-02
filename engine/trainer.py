import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np


def train_one_epoch(model, train_loader, criterion, optimizer, device,
                    scaler=None, grad_clip_norm=None, accum_steps=1):
    """
    One training epoch.

    Args:
        scaler        : torch.amp.GradScaler instance for AMP, or None
        grad_clip_norm: max gradient norm for clipping, or None
        accum_steps   : if > 1, split each DataLoader batch into this many
                        equal micro-batches and accumulate gradients before
                        calling optimizer.step(). Mathematically equivalent
                        to one full-batch pass (preserves fair comparison
                        against models that fit at the full batch).
    """
    model.train()
    running_loss, running_loss_b, running_loss_s = 0.0, 0.0, 0.0
    all_preds_b, all_targets_b = [], []
    all_preds_s, all_targets_s = [], []

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for images, targets_b, targets_s in pbar:
        images    = images.to(device,    non_blocking=True)
        targets_b = targets_b.to(device, non_blocking=True)
        targets_s = targets_s.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Split batch into accum_steps chunks (preserves effective batch size)
        img_chunks = images.chunk(accum_steps) if accum_steps > 1 else [images]
        bt_chunks  = targets_b.chunk(accum_steps) if accum_steps > 1 else [targets_b]
        st_chunks  = targets_s.chunk(accum_steps) if accum_steps > 1 else [targets_s]

        batch_loss = 0.0
        batch_loss_b = 0.0
        batch_loss_s = 0.0
        chunk_pred_b_list, chunk_pred_s_list = [], []

        for ch_img, ch_tb, ch_ts in zip(img_chunks, bt_chunks, st_chunks):
            if scaler is not None:
                with torch.amp.autocast('cuda'):
                    pred_b, pred_s = model(ch_img)
                    loss, loss_b, loss_s = criterion(pred_b, pred_s, ch_tb, ch_ts)
                # Average across micro-batches → equivalent to single full-batch loss
                scaler.scale(loss / accum_steps).backward()
            else:
                pred_b, pred_s = model(ch_img)
                loss, loss_b, loss_s = criterion(pred_b, pred_s, ch_tb, ch_ts)
                (loss / accum_steps).backward()

            batch_loss   += loss.item()   / accum_steps
            batch_loss_b += loss_b.item() / accum_steps
            batch_loss_s += (loss_s.item() if not torch.isnan(loss_s) else 0) / accum_steps

            chunk_pred_b_list.append(pred_b.detach())
            chunk_pred_s_list.append(pred_s.detach())

        if scaler is not None:
            if grad_clip_norm:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            if grad_clip_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        running_loss   += batch_loss
        running_loss_b += batch_loss_b
        running_loss_s += batch_loss_s

        full_pred_b = torch.cat(chunk_pred_b_list, dim=0)
        full_pred_s = torch.cat(chunk_pred_s_list, dim=0)

        all_preds_b.extend(torch.argmax(full_pred_b, dim=1).cpu().numpy())
        all_targets_b.extend(targets_b.cpu().numpy())

        mask = targets_s != -1
        if mask.sum() > 0:
            all_preds_s.extend(torch.argmax(full_pred_s[mask], dim=1).cpu().numpy())
            all_targets_s.extend(targets_s[mask].cpu().numpy())

        pbar.set_postfix({'loss': f'{batch_loss:.4f}'})

    n      = len(train_loader)
    acc_s  = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    return (running_loss / n, running_loss_b / n, running_loss_s / n,
            accuracy_score(all_targets_b, all_preds_b), acc_s)


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds_b, all_targets_b = [], []
    all_preds_s, all_targets_s = [], []

    with torch.no_grad():
        for images, targets_b, targets_s in tqdm(val_loader, desc="Validating", leave=False):
            images    = images.to(device,    non_blocking=True)
            targets_b = targets_b.to(device, non_blocking=True)
            targets_s = targets_s.to(device, non_blocking=True)

            pred_b, pred_s = model(images)
            loss, _, _ = criterion(pred_b, pred_s, targets_b, targets_s)
            running_loss += loss.item()

            all_preds_b.extend(torch.argmax(pred_b, dim=1).cpu().numpy())
            all_targets_b.extend(targets_b.cpu().numpy())

            mask = targets_s != -1
            if mask.sum() > 0:
                all_preds_s.extend(torch.argmax(pred_s[mask], dim=1).cpu().numpy())
                all_targets_s.extend(targets_s[mask].cpu().numpy())

    acc_s = accuracy_score(all_targets_s, all_preds_s) if all_targets_s else 0.0
    return (running_loss / len(val_loader),
            accuracy_score(all_targets_b, all_preds_b),
            acc_s)
