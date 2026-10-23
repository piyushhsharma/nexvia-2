# AIML-02: AI-Assisted Medical Imaging & Diagnosis System

CPU-only medical imaging system with classification, segmentation, and clinical workflow integration.

## Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Prepare data
py scripts/stage1_data_prep.py

# Train models
py scripts/stage2_train_classifier.py
py scripts/stage3_train_segmentation.py

# Generate Grad-CAM heatmaps
py scripts/stage4_gradcam.py
```

## Quickstart

```bash
# Start all services with Docker Compose
# Linux/Mac:
make up

# Windows (without make):
docker-compose up -d

# Access services:
# - Orthanc: http://localhost:8042
# - API: http://localhost:8000
# - Web UI: http://localhost:3000
# - API docs: http://localhost:8000/docs
```

## Results

### Classification (DenseNet121)
- Validation Accuracy: 30.00%
- Per-label AUROC:
  - Pneumonia: 0.812
  - Effusion: 0.357
  - Cardiomegaly: 0.094
  - No Finding: 0.607

### Segmentation (MONAI 3D UNet)
- Validation Dice: 0.6264

### Tests
- 4/4 tests passed (see reports/tests.txt)

## Architecture

See [docs/architecture.md](docs/architecture.md) for system diagram.

## API Documentation

See [docs/api.md](docs/api.md) for API endpoints.

## Limitations

See [docs/limits_and_bias.md](docs/limits_and_bias.md) for dataset bias and model limitations.

## Project Structure

```
medimg/
├── data/           # Raw and processed data
├── models/         # Trained model weights
├── reports/        # Metrics and Grad-CAM outputs
├── src/
│   ├── data/       # Data preprocessing
│   ├── models/     # Model definitions
│   ├── inference/  # Grad-CAM generation
│   ├── api/        # FastAPI backend
│   └── worker/     # Orthanc polling worker
├── web/            # React frontend
├── tests/          # Pytest tests
└── scripts/        # Training and data scripts
```
