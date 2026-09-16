from database.schema import get_db_connection

class PatientRepository:
    def __init__(self, db_path='clinic_local.db'):
        self.db_path = db_path

    def register_patient(self, patient_id, status='ACTIVE', patient_name=''):
        """Registers a new patient in the database."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            name_to_store = patient_name.strip() if patient_name else patient_id
            cursor.execute("PRAGMA table_info(patients)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'patient_name' in cols:
                cursor.execute(
                    "INSERT INTO patients (patient_id, patient_name, status) VALUES (?, ?, ?)",
                    (patient_id, name_to_store, status)
                )
            else:
                cursor.execute(
                    "INSERT INTO patients (patient_id, status) VALUES (?, ?)",
                    (patient_id, status)
                )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error registering patient: {e}")
            return False
        finally:
            conn.close()

    def get_patient(self, patient_id):
        """Retrieves patient record by ID."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        row = cursor.execute(
            "SELECT * FROM patients WHERE patient_id = ?",
            (patient_id,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None

    def patient_exists(self, patient_id):
        """Checks if a patient ID already exists."""
        return self.get_patient(patient_id) is not None

    def list_all_patients(self):
        """Lists all registered patients."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        rows = cursor.execute("SELECT * FROM patients ORDER BY created_at DESC").fetchall()
        conn.close()
        return [dict(r) for r in rows]
