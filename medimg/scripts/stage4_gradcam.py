import torch
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import DenseNetClassifier
from src.inference.gradcam import batch_generate_gradcams


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DenseNetClassifier(num_classes=4)
    model.load_state_dict(torch.load('models/classifier_best.pth', map_location=device))
    
    os.makedirs('reports/gradcam', exist_ok=True)
    
    batch_generate_gradcams(model, 'data/raw/chest_xray/images', 'reports/gradcam')
    
    print("\nStage 4 complete: Grad-CAM heatmaps generated")


if __name__ == '__main__':
    main()
