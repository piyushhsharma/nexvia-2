#!/bin/bash
set -e

echo "Downloading NIH ChestX-ray14 sample..."
mkdir -p data/raw/chest_xray
cd data/raw/chest_xray

# Try to download sample from Kaggle or NIH
if command -v kaggle &> /dev/null; then
    kaggle datasets download -d nih-chest-xrays/data --unzip || echo "Kaggle download failed, using synthetic data"
else
    echo "Kaggle CLI not found, using synthetic data"
fi

# If download failed or empty, generate synthetic data
if [ ! "$(ls -A images)" ]; then
    echo "Generating synthetic chest X-ray data..."
    mkdir -p images
    python -c "
import numpy as np
from PIL import Image
import os
import random

labels = ['Pneumonia', 'Effusion', 'Cardiomegaly', 'No Finding']
os.makedirs('images', exist_ok=True)
os.makedirs('labels', exist_ok=True)

for i in range(100):
    img = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))
    img.save(f'images/img_{i:04d}.png')
    label = random.choice(labels)
    with open(f'labels/img_{i:04d}.txt', 'w') as f:
        f.write(label)
"
fi

cd ../../../

echo "Downloading Medical Segmentation Decathlon Spleen..."
mkdir -p data/raw/spleen
cd data/raw/spleen

if command -v wget &> /dev/null; then
    wget -c https://drive.google.com/uc?id=1RzPB1_bqzE7oTLP6EWU3X7jQlN_djelF -O spleen.tar || echo "Download failed, using synthetic data"
else
    echo "wget not found, using synthetic data"
fi

if [ ! -f spleen.tar ]; then
    echo "Generating synthetic spleen segmentation data..."
    python -c "
import numpy as np
import nibabel as nib
import os

os.makedirs('imagesTr', exist_ok=True)
os.makedirs('labelsTr', exist_ok=True)

for i in range(5):
    # Create small 3D volume
    vol = np.random.randint(0, 255, (32, 32, 32), dtype=np.uint8)
    img = nib.Nifti1Image(vol, np.eye(4))
    nib.save(img, f'imagesTr/spleen_{i:03d}.nii.gz')
    
    # Create segmentation mask
    mask = np.random.randint(0, 2, (32, 32, 32), dtype=np.uint8)
    seg = nib.Nifti1Image(mask, np.eye(4))
    nib.save(seg, f'labelsTr/spleen_{i:03d}.nii.gz')
"
fi

cd ../../../

echo "Data preparation complete."
