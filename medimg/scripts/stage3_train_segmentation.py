import torch
import torch.nn as nn
import torch.optim as optim
from monai.networks.nets import UNet
from monai.losses import DiceLoss
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.segmentation import train_epoch, evaluate
from src.data.preprocessing import SpleenDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = SpleenDataset('data/raw/spleen/imagesTr', 'data/raw/spleen/labelsTr')
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2
    ).to(device)
    
    criterion = DiceLoss(sigmoid=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 10
    best_dice = 0
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), 'models/segmentation_best.pth')
    
    model.load_state_dict(torch.load('models/segmentation_best.pth'))
    val_loss, val_dice = evaluate(model, val_loader, criterion, device)
    
    report = f"""
Segmentation Report
===================
Best Validation Dice: {val_dice:.4f}
Validation Loss: {val_loss:.4f}
"""
    
    with open('reports/segmentation_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nStage 3 complete: Segmentation model trained")


if __name__ == '__main__':
    main()
