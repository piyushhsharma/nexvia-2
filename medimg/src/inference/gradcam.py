import torch
import numpy as np
from PIL import Image
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import DenseNetClassifier
from torchvision import transforms


def generate_gradcam(model, image_path, target_layer, output_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = preprocess(img).unsqueeze(0).to(device)
    
    cam = GradCAM(model=model, target_layers=[target_layer])
    
    targets = None
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    rgb_img = np.array(img.resize((224, 224)))
    rgb_img = np.float32(rgb_img) / 255
    
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    vis_img = Image.fromarray((visualization * 255).astype(np.uint8))
    vis_img.save(output_path)
    
    return grayscale_cam


def batch_generate_gradcams(model, image_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    target_layer = model.model.features[-1]
    
    for img_file in os.listdir(image_dir):
        if img_file.endswith('.png'):
            img_path = os.path.join(image_dir, img_file)
            output_path = os.path.join(output_dir, f'gradcam_{img_file}')
            generate_gradcam(model, img_path, target_layer, output_path)
    
    print(f"Generated Grad-CAM heatmaps in {output_dir}")
