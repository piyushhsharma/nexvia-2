import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models.classifier import DenseNetClassifier


def test_classifier_output_shape():
    model = DenseNetClassifier(num_classes=4)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (2, 4)
