import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from PIL import Image
import random
from src.data.dicom_converter import convert_batch
from src.data.preprocessing import normalize_image


def generate_synthetic_chest_xray():
    print("Generating synthetic chest X-ray data...")
    os.makedirs('data/raw/chest_xray/images', exist_ok=True)
    os.makedirs('data/raw/chest_xray/labels', exist_ok=True)
    
    labels = ['Pneumonia', 'Effusion', 'Cardiomegaly', 'No Finding']
    
    for i in range(100):
        img = Image.fromarray(np.random.randint(50, 200, (224, 224), dtype=np.uint8))
        img.save(f'data/raw/chest_xray/images/img_{i:04d}.png')
        label = random.choice(labels)
        with open(f'data/raw/chest_xray/labels/img_{i:04d}.txt', 'w') as f:
            f.write(label)
    
    print(f"Generated 100 synthetic chest X-ray images")


def generate_synthetic_spleen():
    print("Generating synthetic spleen segmentation data...")
    import nibabel as nib
    
    os.makedirs('data/raw/spleen/imagesTr', exist_ok=True)
    os.makedirs('data/raw/spleen/labelsTr', exist_ok=True)
    
    for i in range(5):
        vol = np.random.randint(0, 255, (64, 64, 64), dtype=np.uint8)
        img = nib.Nifti1Image(vol, np.eye(4))
        nib.save(img, f'data/raw/spleen/imagesTr/spleen_{i:03d}.nii.gz')
        
        mask = np.random.randint(0, 2, (64, 64, 64), dtype=np.uint8)
        seg = nib.Nifti1Image(mask, np.eye(4))
        nib.save(seg, f'data/raw/spleen/labelsTr/spleen_{i:03d}.nii.gz')
    
    print(f"Generated 5 synthetic spleen volumes")


def create_train_test_split():
    print("Creating train/test split...")
    images = sorted([f for f in os.listdir('data/raw/chest_xray/images') if f.endswith('.png')])
    random.shuffle(images)
    
    train_split = int(0.8 * len(images))
    train_images = images[:train_split]
    test_images = images[train_split:]
    
    os.makedirs('data/processed', exist_ok=True)
    
    with open('data/processed/train.txt', 'w') as f:
        for img in train_images:
            label_file = img.replace('.png', '.txt')
            with open(f'data/raw/chest_xray/labels/{label_file}', 'r') as lf:
                label = lf.read().strip()
            f.write(f'{img} {label}\n')
    
    with open('data/processed/test.txt', 'w') as f:
        for img in test_images:
            label_file = img.replace('.png', '.txt')
            with open(f'data/raw/chest_xray/labels/{label_file}', 'r') as lf:
                label = lf.read().strip()
            f.write(f'{img} {label}\n')
    
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")


def convert_to_dicom():
    print("Converting PNG to DICOM...")
    os.makedirs('data/dicom', exist_ok=True)
    convert_batch('data/raw/chest_xray/images', 'data/dicom')
    print(f"Converted DICOM files to data/dicom")


if __name__ == '__main__':
    generate_synthetic_chest_xray()
    generate_synthetic_spleen()
    create_train_test_split()
    convert_to_dicom()
    print("\nStage 1 complete: Data preparation done")
