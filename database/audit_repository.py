from database.schema import get_db_connection

class AuditRepository:
    def __init__(self, db_path='clinic_local.db'):
        self.db_path = db_path

    def log_event(self, event_type, actor, patient_id=None, visit_id=None):
        """Records an event in the local audit logs table."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO audit_logs (event_type, actor, patient_id, visit_id) 
                VALUES (?, ?, ?, ?)""",
                (event_type, actor, patient_id, visit_id)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error logging audit event: {e}")
            return False
        finally:
            conn.close()

    def get_logs(self, limit=100):
        """Retrieves list of audit log records."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        rows = cursor.execute(
            "SELECT * FROM audit_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
