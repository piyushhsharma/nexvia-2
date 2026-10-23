import { useState, useEffect } from 'react'
import './AuditLog.css'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function AuditLog() {
  const [audits, setAudits] = useState([])

  useEffect(() => {
    fetchAudits()
  }, [])

  const fetchAudits = async () => {
    try {
      const res = await fetch(`${API_URL}/audit`)
      const data = await res.json()
      setAudits(data)
    } catch (err) {
      console.error('Failed to fetch audit log:', err)
    }
  }

  return (
    <div className="audit-log">
      <h2>Audit Log</h2>
      {audits.length === 0 ? (
        <p>No audit entries</p>
      ) : (
        <table className="audit-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>User</th>
              <th>Study ID</th>
              <th>Decision</th>
              <th>Comment</th>
              <th>AI Output</th>
              <th>Model Version</th>
            </tr>
          </thead>
          <tbody>
            {audits.map(audit => (
              <tr key={audit.id}>
                <td>{new Date(audit.timestamp).toLocaleString()}</td>
                <td>{audit.user}</td>
                <td>{audit.study_id}</td>
                <td className={`decision ${audit.decision.toLowerCase()}`}>{audit.decision}</td>
                <td>{audit.comment || '-'}</td>
                <td>{audit.ai_output}</td>
                <td>{audit.model_version}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default AuditLog
