# API Documentation

Base URL: `http://localhost:8000`

## Endpoints

### GET /studies
Returns all studies with AI predictions.

**Response:**
```json
[
  {
    "id": 1,
    "study_id": "abc123",
    "patient_id": "PAT001",
    "prediction": "Pneumonia",
    "confidence": 0.85,
    "heatmap_path": "reports/gradcam/gradcam_abc123.png",
    "segmentation_path": null,
    "status": "pending review",
    "created_at": "2026-10-22T10:00:00"
  }
]
```

### GET /studies/{id}
Returns a single study by ID.

### POST /studies/{id}/decision
Submit approve/reject decision for a study.

**Request:**
```json
{
  "decision": "Approved",
  "comment": "Agrees with AI assessment"
}
```

**Response:**
```json
{
  "status": "success",
  "decision": "Approved"
}
```

### GET /audit
Returns append-only audit log of all decisions.

**Response:**
```json
[
  {
    "id": 1,
    "study_id": 1,
    "user": "radiologist",
    "decision": "Approved",
    "comment": "Agrees with AI assessment",
    "model_version": "v1.0",
    "ai_output": "Pneumonia",
    "timestamp": "2026-10-22T10:05:00"
  }
]
```

### GET /heatmaps/{filename}
Returns Grad-CAM heatmap image.

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```
