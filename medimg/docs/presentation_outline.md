# Presentation Outline (10 minutes)

## Problem (2 minutes)
- Medical imaging volume increasing faster than radiologist capacity
- Need for AI assistance to prioritize and triage studies
- Challenge: building trustworthy, auditable clinical AI systems

## Design (2 minutes)
- System architecture: Orthanc → Worker → API → Web UI
- Two tasks: multi-label classification (DenseNet121) + 3D segmentation (MONAI UNet)
- Explainability: Grad-CAM heatmaps for model transparency
- Clinical workflow: approve/reject decisions with audit trail

## Challenges (2 minutes)
- CPU-only constraint: small models, limited training epochs
- Data availability: synthetic data due to download failures
- Docker integration: Orthanc PACS, worker polling, API coordination
- Real-time inference vs. batch processing tradeoffs

## Results (2 minutes)
- Classification: 25% validation accuracy, AUROCs [0.347, 0.267, 0.643, 0.547]
- Segmentation: 0.6268 Dice score on synthetic spleen data
- System fully functional: data pipeline, training, inference, API, web UI
- All tests passing (4/4), GitHub Actions CI configured

## Next Steps (2 minutes)
- Real clinical data integration (NIH ChestX-ray14 full dataset)
- GPU training for larger models and more epochs
- External validation on held-out hospital data
- Regulatory pathway exploration (FDA De Novo)
- User studies with radiologists for workflow validation
