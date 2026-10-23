import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import nibabel as nib
import os


class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, label_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.label_map = {'Pneumonia': 0, 'Effusion': 1, 'Cardiomegaly': 2, 'No Finding': 3}
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = parts[1]
                    self.images.append(img_name)
                    self.labels.append(self.label_map.get(label, 3))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.data_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, torch.tensor(label, dtype=torch.long)


def normalize_image(img):
    img = np.array(img, dtype=np.float32)
    img = img / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img


def crop_volume(volume, crop_size=(64, 64, 64)):
    _, h, w, d = volume.shape
    ch, cw, cd = crop_size
    
    start_h = max(0, (h - ch) // 2)
    start_w = max(0, (w - cw) // 2)
    start_d = max(0, (d - cd) // 2)
    
    end_h = min(h, start_h + ch)
    end_w = min(w, start_w + cw)
    end_d = min(d, start_d + cd)
    
    cropped = volume[:, start_h:end_h, start_w:end_w, start_d:end_d]
    
    if cropped.shape[1:] != crop_size:
        pad_h = ch - cropped.shape[1]
        pad_w = cw - cropped.shape[2]
        pad_d = cd - cropped.shape[3]
        cropped = np.pad(cropped, ((0, 0), (0, pad_h), (0, pad_w), (0, pad_d)), mode='constant')
    
    return cropped


class SpleenDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.nii.gz')])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_file)
        label_path = os.path.join(self.labels_dir, img_file)
        
        img = nib.load(img_path).get_fdata()
        label = nib.load(label_path).get_fdata()
        
        img = np.expand_dims(img, axis=0)
        label = np.expand_dims(label, axis=0)
        
        img = crop_volume(img)
        label = crop_volume(label)
        
        img = torch.from_numpy(img).float()
        label = torch.from_numpy(label).float()
        
        if self.transform:
            img, label = self.transform(img, label)
        
        return img, label
