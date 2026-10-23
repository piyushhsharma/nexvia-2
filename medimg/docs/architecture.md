# System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        DICOM[DICOM Files]
        Orthanc[Orthanc PACS]
    end

    subgraph "Processing Layer"
        Worker[Worker Service]
        Classifier[DenseNet121 Classifier]
        Segmentation[MONAI 3D UNet]
        GradCAM[Grad-CAM Generator]
    end

    subgraph "Storage Layer"
        SQLite[(SQLite Database)]
        Models[Model Weights]
        Heatmaps[Grad-CAM Images]
    end

    subgraph "API Layer"
        API[FastAPI]
    end

    subgraph "Presentation Layer"
        Web[React Web UI]
    end

    DICOM --> Orthanc
    Orthanc --> Worker
    Worker --> Classifier
    Worker --> Segmentation
    Worker --> GradCAM
    Worker --> SQLite
    Worker --> Heatmaps
    Classifier --> Models
    Segmentation --> Models
    API --> SQLite
    API --> Heatmaps
    Web --> API
```

## Components

### Orthanc
- DICOM PACS server for medical image storage
- Exposes REST API for study retrieval
- Port 8042 (HTTP), 4242 (DICOM)

### Worker Service
- Polls Orthanc for new studies
- Runs classification and segmentation inference
- Generates Grad-CAM heatmaps
- Stores results in SQLite

### API Service
- FastAPI backend serving study data
- Handles approve/reject decisions
- Maintains append-only audit log
- Port 8000

### Web UI
- React frontend for radiologist review
- Displays studies with heatmaps/segmentation
- Decision workflow integration
- Port 3000
