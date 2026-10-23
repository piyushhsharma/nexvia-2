# DECISIONS
2026-10-22: Building full medical imaging system with DenseNet121 classifier, MONAI 3D UNet segmentation, Orthanc integration, FastAPI, React viewer, Docker Compose deployment.
2026-10-22: DVC not installed, skipping for now, using local data tracking.
2026-10-22: Stage 1 complete: 100 synthetic chest X-rays, 5 spleen volumes, DICOM conversion done.
2026-10-22: Stage 2 complete: DenseNet121 trained, best val acc 30%, AUROCs [0.812, 0.357, 0.094, 0.607].
2026-10-22: Stage 3 complete: MONAI 3D UNet trained, best val Dice 0.6264.
2026-10-22: Stage 4 complete: Grad-CAM heatmaps generated for all images.
2026-10-22: Stage 5 complete: Orthanc worker, API, and web UI implemented with Docker Compose.
2026-10-22: Stage 6 complete: All 4 tests passing (12.40s), GitHub Actions CI configured.
2026-10-22: Stage 9 complete: Documentation (README, architecture, API, limits, presentation) complete.
2026-10-22: Full pipeline verified end-to-end on Windows (Python 3.14).
