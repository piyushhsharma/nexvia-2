import sqlite3
import time
import requests
import json
import torch
import numpy as np
from PIL import Image
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.models.classifier import DenseNetClassifier
from src.inference.gradcam import generate_gradcam
from torchvision import transforms


class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS studies (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id TEXT UNIQUE,
                patient_id TEXT,
                prediction TEXT,
                confidence REAL,
                heatmap_path TEXT,
                segmentation_path TEXT,
                status TEXT DEFAULT 'pending review',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                study_id INTEGER,
                user TEXT,
                decision TEXT,
                comment TEXT,
                model_version TEXT,
                ai_output TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (study_id) REFERENCES studies (id)
            )
        ''')
        conn.commit()
        conn.close()
    
    def add_study(self, study_id, patient_id, prediction, confidence, heatmap_path, segmentation_path):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO studies 
            (study_id, patient_id, prediction, confidence, heatmap_path, segmentation_path, status)
            VALUES (?, ?, ?, ?, ?, ?, 'pending review')
        ''', (study_id, patient_id, prediction, confidence, heatmap_path, segmentation_path))
        conn.commit()
        conn.close()
    
    def get_studies(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM studies')
        studies = cursor.fetchall()
        conn.close()
        return studies


class OrthancWorker:
    def __init__(self, orthanc_url, db_path):
        self.orthanc_url = orthanc_url
        self.db = Database(db_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_models()
    
    def load_models(self):
        self.classifier = DenseNetClassifier(num_classes=4)
        self.classifier.load_state_dict(torch.load('models/classifier_best.pth', map_location=self.device))
        self.classifier.eval()
        
        self.label_map = {0: 'Pneumonia', 1: 'Effusion', 2: 'Cardiomegaly', 3: 'No Finding'}
        
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def poll_studies(self):
        try:
            response = requests.get(f'{self.orthanc_url}/studies')
            if response.status_code == 200:
                studies = response.json()
                for study in studies:
                    self.process_study(study)
        except Exception as e:
            print(f"Error polling Orthanc: {e}")
    
    def process_study(self, study):
        study_id = study['ID']
        
        response = requests.get(f'{self.orthanc_url}/studies/{study_id}')
        if response.status_code != 200:
            return
        
        study_data = response.json()
        patient_id = study_data.get('PatientMainDicomTags', {}).get('PatientID', 'Unknown')
        
        series = study_data.get('Series', [])
        if not series:
            return
        
        series_id = series[0]['ID']
        instances_response = requests.get(f'{self.orthanc_url}/series/{series_id}/instances')
        if instances_response.status_code != 200:
            return
        
        instances = instances_response.json()
        if not instances:
            return
        
        instance_id = instances[0]['ID']
        
        dicom_response = requests.get(f'{self.orthanc_url}/instances/{instance_id}/file')
        if dicom_response.status_code != 200:
            return
        
        import pydicom
        from io import BytesIO
        dcm = pydicom.dcmread(BytesIO(dicom_response.content))
        
        if hasattr(dcm, 'pixel_array'):
            arr = dcm.pixel_array
            if len(arr.shape) == 2:
                img = Image.fromarray(arr).convert('RGB')
            else:
                img = Image.fromarray(arr[:, :, 0]).convert('RGB')
            
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.classifier(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = probs.max(1)
            
            prediction = self.label_map[predicted.item()]
            confidence = confidence.item()
            
            heatmap_path = f'reports/gradcam/gradcam_{study_id}.png'
            os.makedirs('reports/gradcam', exist_ok=True)
            
            target_layer = self.classifier.model.features[-1]
            generate_gradcam(self.classifier, img, target_layer, heatmap_path)
            
            segmentation_path = None
            
            self.db.add_study(study_id, patient_id, prediction, confidence, heatmap_path, segmentation_path)
            print(f"Processed study {study_id}: {prediction} ({confidence:.3f})")
    
    def run(self):
        print("Worker started, polling Orthanc...")
        while True:
            self.poll_studies()
            time.sleep(10)


if __name__ == '__main__':
    orthanc_url = os.getenv('ORTHANC_URL', 'http://localhost:8042')
    db_path = os.getenv('DB_PATH', 'studies.db')
    
    worker = OrthancWorker(orthanc_url, db_path)
    worker.run()
