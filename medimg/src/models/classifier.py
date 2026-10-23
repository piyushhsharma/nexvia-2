import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data.preprocessing import ChestXrayDataset


class DenseNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.densenet121(weights='IMAGENET1K_V1')
        num_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.model(x)


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def compute_class_weights(labels, num_classes=4):
    class_counts = np.bincount(labels, minlength=num_classes)
    total = len(labels)
    weights = total / (num_classes * class_counts + 1e-6)
    return torch.FloatTensor(weights)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    accuracy = 100. * (all_preds == all_labels).sum() / len(all_labels)
    
    aurocs = []
    for i in range(4):
        binary_labels = (all_labels == i).astype(int)
        if binary_labels.sum() > 0:
            auroc = roc_auc_score(binary_labels, all_probs[:, i])
            aurocs.append(auroc)
    
    cm = confusion_matrix(all_labels, all_preds)
    
    sensitivity = []
    specificity = []
    for i in range(4):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        
        sens = tp / (tp + fn + 1e-6)
        spec = tn / (tn + fp + 1e-6)
        sensitivity.append(sens)
        specificity.append(spec)
    
    return total_loss / len(loader), accuracy, aurocs, sensitivity, specificity, cm
