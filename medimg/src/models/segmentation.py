import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import SpleenDataset


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_dice = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        with torch.no_grad():
            preds = (outputs > 0.5).float()
            dice = (2 * (preds * labels).sum()) / (preds.sum() + labels.sum() + 1e-6)
            total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_dice = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            preds = (outputs > 0.5).float()
            dice = (2 * (preds * labels).sum()) / (preds.sum() + labels.sum() + 1e-6)
            total_dice += dice.item()
    
    return total_loss / len(loader), total_dice / len(loader)
