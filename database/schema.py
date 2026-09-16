import sqlite3
import os

def get_db_connection(db_path='clinic_local.db'):
    conn = sqlite3.connect(db_path)
    # Enable foreign keys support in SQLite
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn

def init_database(db_path='clinic_local.db'):
    """Initializes all required tables in the local SQLite database."""
    conn = get_db_connection(db_path)
    cursor = conn.cursor()
    
    # 1. PATIENTS Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id TEXT PRIMARY KEY,
        patient_name TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        status TEXT DEFAULT 'ACTIVE'
    )
    """)
    cursor.execute("PRAGMA table_info(patients)")
    patient_cols = [row[1] for row in cursor.fetchall()]
    if 'patient_name' not in patient_cols:
        cursor.execute("ALTER TABLE patients ADD COLUMN patient_name TEXT DEFAULT ''")
    
    # 2. VISITS Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS visits (
        visit_id TEXT PRIMARY KEY,
        patient_id TEXT,
        visit_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        clinic_id TEXT,
        model_version TEXT,
        FOREIGN KEY (patient_id) REFERENCES patients(patient_id) ON DELETE CASCADE
    )
    """)
    
    # 3. CLINICAL_OBSERVATION Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinical_observations (
        observation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT,
        age REAL,
        gender TEXT,
        diabetes INTEGER,
        passive_smoke_exposure INTEGER,
        family_respiratory_history INTEGER,
        height REAL DEFAULT 0,
        weight REAL DEFAULT 0,
        blood_pressure TEXT DEFAULT '',
        blood_sugar REAL DEFAULT 0,
        data_type TEXT DEFAULT 'synthetic',
        synthetic INTEGER DEFAULT 1,
        FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE
    )
    """)
    cursor.execute("PRAGMA table_info(clinical_observations)")
    obs_cols = [row[1] for row in cursor.fetchall()]
    for col_name, col_type in [('height', 'REAL DEFAULT 0'), ('weight', 'REAL DEFAULT 0'), ('blood_pressure', 'TEXT DEFAULT ""'), ('blood_sugar', 'REAL DEFAULT 0')]:
        if col_name not in obs_cols:
            cursor.execute(f"ALTER TABLE clinical_observations ADD COLUMN {col_name} {col_type}")
    
    # 4. INFERENCE Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS inferences (
        inference_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT,
        predicted_class INTEGER,
        pneumonia_probability REAL,
        confidence REAL,
        uncertainty REAL,
        inference_latency REAL,
        FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE
    )
    """)
    
    # 5. XAI_RESULT Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS xai_results (
        xai_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT,
        method TEXT,
        heatmap_reference TEXT,
        FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE
    )
    """)
    
    # 6. CLINICIAN_REVIEW Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS clinician_reviews (
        review_id INTEGER PRIMARY KEY AUTOINCREMENT,
        visit_id TEXT,
        review_status TEXT CHECK(review_status IN ('CONFIRMED', 'NEEDS_REVIEW', 'UNABLE_TO_DETERMINE')),
        review_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        clinician_reference TEXT,
        clinical_note TEXT,
        FOREIGN KEY (visit_id) REFERENCES visits(visit_id) ON DELETE CASCADE
    )
    """)
    
    # 7. AUDIT_LOG Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS audit_logs (
        audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        event_type TEXT,
        actor TEXT,
        patient_id TEXT,
        visit_id TEXT
    )
    """)
    
    # Create Indexes for fast querying
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_observations_visit ON clinical_observations(visit_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_inferences_visit ON inferences(visit_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_audit_patient ON audit_logs(patient_id)")
    
    conn.commit()
    conn.close()
    print(f"Local clinic database initialized at: {db_path}")

if __name__ == '__main__':
    init_database()
