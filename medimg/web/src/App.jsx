import { useState, useEffect } from 'react'
import StudyList from './components/StudyList'
import StudyViewer from './components/StudyViewer'
import AuditLog from './components/AuditLog'
import './App.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [studies, setStudies] = useState([])
  const [selectedStudy, setSelectedStudy] = useState(null)
  const [view, setView] = useState('list')
  const [showHeatmap, setShowHeatmap] = useState(false)
  const [showSegmentation, setShowSegmentation] = useState(false)

  useEffect(() => {
    fetchStudies()
  }, [])

  const fetchStudies = async () => {
    try {
      const res = await fetch(`${API_URL}/studies`)
      const data = await res.json()
      setStudies(data)
    } catch (err) {
      console.error('Failed to fetch studies:', err)
    }
  }

  const handleDecision = async (studyId, decision, comment) => {
    try {
      await fetch(`${API_URL}/studies/${studyId}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ decision, comment })
      })
      fetchStudies()
      setSelectedStudy(null)
      setView('list')
    } catch (err) {
      console.error('Failed to submit decision:', err)
    }
  }

  return (
    <div className="app">
      <div className="banner">
        For education only. Not for clinical use.
      </div>
      
      <div className="header">
        <h1>MedImg AI Assistant</h1>
        <nav>
          <button onClick={() => setView('list')} className={view === 'list' ? 'active' : ''}>Studies</button>
          <button onClick={() => setView('audit')} className={view === 'audit' ? 'active' : ''}>Audit Log</button>
        </nav>
      </div>

      {view === 'list' && (
        <div className="content">
          {selectedStudy ? (
            <StudyViewer
              study={selectedStudy}
              showHeatmap={showHeatmap}
              showSegmentation={showSegmentation}
              onToggleHeatmap={() => setShowHeatmap(!showHeatmap)}
              onToggleSegmentation={() => setShowSegmentation(!showSegmentation)}
              onDecision={handleDecision}
              onBack={() => setSelectedStudy(null)}
            />
          ) : (
            <StudyList studies={studies} onSelectStudy={setSelectedStudy} />
          )}
        </div>
      )}

      {view === 'audit' && (
        <div className="content">
          <AuditLog />
        </div>
      )}
    </div>
  )
}

export default App
