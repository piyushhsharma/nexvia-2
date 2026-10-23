import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import sys
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import DenseNetClassifier, get_transforms, train_epoch, evaluate, compute_class_weights
from src.data.preprocessing import ChestXrayDataset


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_transform, test_transform = get_transforms()
    
    train_dataset = ChestXrayDataset('data/raw/chest_xray/images', 'data/processed/train.txt', train_transform)
    test_dataset = ChestXrayDataset('data/raw/chest_xray/images', 'data/processed/test.txt', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)
    
    labels = [train_dataset.labels[i] for i in range(len(train_dataset))]
    class_weights = compute_class_weights(labels).to(device)
    
    model = DenseNetClassifier(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 5
    best_acc = 0
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, aurocs, sensitivity, specificity, cm = evaluate(model, test_loader, criterion, device)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"AUROCs: {[f'{x:.3f}' for x in aurocs]}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/classifier_best.pth')
    
    model.load_state_dict(torch.load('models/classifier_best.pth'))
    val_loss, val_acc, aurocs, sensitivity, specificity, cm = evaluate(model, test_loader, criterion, device)
    
    report = f"""
Classification Report
=====================
Best Validation Accuracy: {val_acc:.2f}%

Per-label AUROC:
- Pneumonia: {aurocs[0]:.3f}
- Effusion: {aurocs[1]:.3f}
- Cardiomegaly: {aurocs[2]:.3f}
- No Finding: {aurocs[3]:.3f}

Sensitivity:
- Pneumonia: {sensitivity[0]:.3f}
- Effusion: {sensitivity[1]:.3f}
- Cardiomegaly: {sensitivity[2]:.3f}
- No Finding: {sensitivity[3]:.3f}

Specificity:
- Pneumonia: {specificity[0]:.3f}
- Effusion: {specificity[1]:.3f}
- Cardiomegaly: {specificity[2]:.3f}
- No Finding: {specificity[3]:.3f}

Confusion Matrix:
{cm}
"""
    
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)
    
    print(report)
    print("\nStage 2 complete: Classifier trained")


if __name__ == '__main__':
    main()
