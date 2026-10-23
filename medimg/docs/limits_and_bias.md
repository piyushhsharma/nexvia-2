# Limitations and Bias

## Dataset Bias

### Synthetic Data
- Current system uses 100 synthetic chest X-ray images
- Not representative of real clinical data distribution
- Limited label variety (4 classes only)
- No demographic information included

### Small Sample Size
- 5 spleen volumes for segmentation training
- 80/20 train/test split on 100 images
- Insufficient for robust model generalization
- High variance in performance metrics

## Label Noise

### Synthetic Labels
- Randomly assigned labels in synthetic data
- No radiologist verification
- May not reflect true clinical presentation
- No inter-rater reliability assessment

## Model Limitations

### CPU-Only Training
- Models trained on CPU only, limited epochs
- DenseNet121: 5 epochs, 25% validation accuracy
- UNet: 10 epochs, 0.6268 Dice score
- Not optimized for production deployment

### Transfer Learning Mismatch
- ImageNet weights for medical images
- Domain gap between natural and medical images
- May not capture relevant medical features

## No External Validation

### Internal Testing Only
- No external dataset validation
- No cross-hospital evaluation
- No comparison with clinical benchmarks
- Performance may not generalize

## Clinical Considerations

### Not for Clinical Use
- System is for educational purposes only
- No regulatory approval (FDA, CE)
- No clinical trials conducted
- Should not inform patient care decisions

### Responsibility
- Radiologist must review all AI predictions
- AI serves as decision support, not replacement
- Human-in-the-loop workflow required
- Audit trail maintained for accountability
