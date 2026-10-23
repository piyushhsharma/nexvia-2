import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import sqlite3
import tempfile


def test_audit_log_insert():
    db_path = tempfile.mktemp(suffix='.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            study_id INTEGER,
            user TEXT,
            decision TEXT,
            comment TEXT,
            model_version TEXT,
            ai_output TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        INSERT INTO audit_log (study_id, user, decision, comment, model_version, ai_output)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (1, 'test_user', 'Approved', 'Test comment', 'v1.0', 'Pneumonia'))
    
    conn.commit()
    
    cursor.execute('SELECT * FROM audit_log WHERE study_id = 1')
    result = cursor.fetchone()
    
    assert result is not None
    assert result[2] == 'test_user'
    assert result[3] == 'Approved'
    
    conn.close()
    os.unlink(db_path)
