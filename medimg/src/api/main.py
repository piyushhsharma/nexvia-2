from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sqlite3
import os
from typing import List, Optional
from datetime import datetime

app = FastAPI(title="MedImg API")

DB_PATH = os.getenv('DB_PATH', 'studies.db')


class Study(BaseModel):
    id: int
    study_id: str
    patient_id: str
    prediction: str
    confidence: float
    heatmap_path: Optional[str]
    segmentation_path: Optional[str]
    status: str
    created_at: str


class DecisionRequest(BaseModel):
    decision: str
    comment: Optional[str] = None


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@app.get("/studies", response_model=List[Study])
def get_studies():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM studies ORDER BY created_at DESC')
    studies = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return studies


@app.get("/studies/{study_id}", response_model=Study)
def get_study(study_id: int):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM studies WHERE id = ?', (study_id,))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Study not found")
    
    return dict(row)


@app.post("/studies/{study_id}/decision")
def make_decision(study_id: int, request: DecisionRequest, user: str = "radiologist"):
    conn = get_db()
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM studies WHERE id = ?', (study_id,))
    study = cursor.fetchone()
    
    if not study:
        conn.close()
        raise HTTPException(status_code=404, detail="Study not found")
    
    cursor.execute('''
        INSERT INTO audit_log (study_id, user, decision, comment, model_version, ai_output)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (study_id, user, request.decision, request.comment, "v1.0", study['prediction']))
    
    cursor.execute('UPDATE studies SET status = ? WHERE id = ?', 
                  (f"{request.decision}", study_id))
    
    conn.commit()
    conn.close()
    
    return {"status": "success", "decision": request.decision}


@app.get("/audit")
def get_audit_log():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM audit_log ORDER BY timestamp DESC')
    audits = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return audits


@app.get("/heatmaps/{filename}")
def get_heatmap(filename: str):
    filepath = os.path.join('reports/gradcam', filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Heatmap not found")
    return FileResponse(filepath)


@app.get("/health")
def health_check():
    return {"status": "healthy"}
