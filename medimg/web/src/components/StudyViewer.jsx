import { useState } from 'react'
import './StudyViewer.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function StudyViewer({ study, showHeatmap, showSegmentation, onToggleHeatmap, onToggleSegmentation, onDecision, onBack }) {
  const [comment, setComment] = useState('')

  const handleSubmit = (decision) => {
    onDecision(study.id, decision, comment)
  }

  return (
    <div className="study-viewer">
      <button onClick={onBack} className="back-button">← Back to List</button>
      
      <div className="viewer-header">
        <h2>Patient: {study.patient_id}</h2>
        <div className="prediction-box">
          <span className="prediction-label">AI Prediction:</span>
          <span className="prediction-value">{study.prediction}</span>
          <span className="confidence-value">({(study.confidence * 100).toFixed(1)}% confidence)</span>
        </div>
      </div>

      <div className="viewer-content">
        <div className="image-viewer">
          <div className="image-container">
            <h3>Scan View</h3>
            <div className="placeholder-image">
              <p>Image would be displayed here</p>
              {showHeatmap && study.heatmap_path && (
                <img 
                  src={`${API_URL}/heatmaps/${study.heatmap_path.split('/').pop()}`} 
                  alt="Heatmap" 
                  className="overlay-image"
                />
              )}
            </div>
          </div>

          <div className="controls">
            <button 
              onClick={onToggleHeatmap}
              className={showHeatmap ? 'active' : ''}
            >
              {showHeatmap ? 'Hide Heatmap' : 'Show Heatmap'}
            </button>
            <button 
              onClick={onToggleSegmentation}
              className={showSegmentation ? 'active' : ''}
              disabled={!study.segmentation_path}
            >
              {showSegmentation ? 'Hide Segmentation' : 'Show Segmentation'}
            </button>
          </div>
        </div>

        <div className="decision-panel">
          <h3>Review Decision</h3>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Add comments (optional)"
            className="comment-input"
          />
          <div className="decision-buttons">
            <button onClick={() => handleSubmit('Approved')} className="approve-button">
              Approve
            </button>
            <button onClick={() => handleSubmit('Rejected')} className="reject-button">
              Reject
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default StudyViewer
