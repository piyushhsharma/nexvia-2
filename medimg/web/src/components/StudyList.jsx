import './StudyList.css'

function StudyList({ studies, onSelectStudy }) {
  return (
    <div className="study-list">
      <h2>Studies</h2>
      {studies.length === 0 ? (
        <p>No studies available</p>
      ) : (
        <div className="studies-grid">
          {studies.map(study => (
            <div key={study.id} className="study-card" onClick={() => onSelectStudy(study)}>
              <h3>Patient: {study.patient_id}</h3>
              <p>Prediction: <strong>{study.prediction}</strong></p>
              <p>Confidence: {(study.confidence * 100).toFixed(1)}%</p>
              <p>Status: <span className={`status ${study.status}`}>{study.status}</span></p>
              <p className="timestamp">{new Date(study.created_at).toLocaleString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default StudyList
