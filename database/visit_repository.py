from database.schema import get_db_connection

class VisitRepository:
    def __init__(self, db_path='clinic_local.db'):
        self.db_path = db_path

    def create_visit(self, visit_id, patient_id, clinic_id, model_version):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO visits (visit_id, patient_id, clinic_id, model_version) VALUES (?, ?, ?, ?)",
                (visit_id, patient_id, clinic_id, model_version)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error creating visit: {e}")
            return False
        finally:
            conn.close()

    def store_clinical_observation(self, visit_id, age, gender, diabetes, smoke, family, data_type='synthetic', synthetic=1, height=0.0, weight=0.0, blood_pressure='', blood_sugar=0.0):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("PRAGMA table_info(clinical_observations)")
            cols = [r[1] for r in cursor.fetchall()]
            if 'blood_pressure' in cols:
                cursor.execute(
                    """INSERT INTO clinical_observations 
                    (visit_id, age, gender, diabetes, passive_smoke_exposure, family_respiratory_history, height, weight, blood_pressure, blood_sugar, data_type, synthetic) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (visit_id, age, gender, diabetes, smoke, family, height or 0.0, weight or 0.0, blood_pressure or '', blood_sugar or 0.0, data_type, synthetic)
                )
            else:
                cursor.execute(
                    """INSERT INTO clinical_observations 
                    (visit_id, age, gender, diabetes, passive_smoke_exposure, family_respiratory_history, data_type, synthetic) 
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (visit_id, age, gender, diabetes, smoke, family, data_type, synthetic)
                )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing observation: {e}")
            return False
        finally:
            conn.close()

    def store_inference(self, visit_id, predicted_class, prob, confidence, uncertainty, latency):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                """INSERT INTO inferences 
                (visit_id, predicted_class, pneumonia_probability, confidence, uncertainty, inference_latency) 
                VALUES (?, ?, ?, ?, ?, ?)""",
                (visit_id, predicted_class, prob, confidence, uncertainty, latency)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing inference: {e}")
            return False
        finally:
            conn.close()

    def store_xai_result(self, visit_id, method, heatmap_ref):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute(
                "INSERT INTO xai_results (visit_id, method, heatmap_reference) VALUES (?, ?, ?)",
                (visit_id, method, heatmap_ref)
            )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing XAI result: {e}")
            return False
        finally:
            conn.close()

    def store_clinician_review(self, visit_id, review_status, clinician_ref, clinical_note):
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        try:
            # Check if review already exists for this visit, if so overwrite/update it
            existing = cursor.execute("SELECT review_id FROM clinician_reviews WHERE visit_id = ?", (visit_id,)).fetchone()
            if existing:
                cursor.execute(
                    """UPDATE clinician_reviews 
                    SET review_status = ?, clinician_reference = ?, clinical_note = ?, review_timestamp = CURRENT_TIMESTAMP 
                    WHERE visit_id = ?""",
                    (review_status, clinician_ref, clinical_note, visit_id)
                )
            else:
                cursor.execute(
                    """INSERT INTO clinician_reviews (visit_id, review_status, clinician_reference, clinical_note) 
                    VALUES (?, ?, ?, ?)""",
                    (visit_id, review_status, clinician_ref, clinical_note)
                )
            conn.commit()
            return True
        except Exception as e:
            print(f"Error storing clinician review: {e}")
            return False
        finally:
            conn.close()

    def get_visit_history(self, patient_id):
        """Retrieves complete timeline of visits and related inferences for a patient."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT v.visit_id, v.visit_timestamp, v.model_version, v.clinic_id,
               o.age, o.gender, o.diabetes, o.passive_smoke_exposure, o.family_respiratory_history,
               o.height, o.weight, o.blood_pressure, o.blood_sugar, o.synthetic,
               i.predicted_class, i.pneumonia_probability, i.confidence, i.uncertainty, i.inference_latency,
               cr.review_status, cr.clinical_note
        FROM visits v
        LEFT JOIN clinical_observations o ON v.visit_id = o.visit_id
        LEFT JOIN inferences i ON v.visit_id = i.visit_id
        LEFT JOIN clinician_reviews cr ON v.visit_id = cr.visit_id
        WHERE v.patient_id = ?
        ORDER BY v.visit_timestamp ASC
        """
        rows = cursor.execute(query, (patient_id,)).fetchall()
        conn.close()
        return [dict(r) for r in rows]

    def get_visit_detail(self, visit_id):
        """Retrieves full details for a single visit."""
        conn = get_db_connection(self.db_path)
        cursor = conn.cursor()
        
        query = """
        SELECT v.visit_id, v.patient_id, v.visit_timestamp, v.model_version, v.clinic_id,
               o.age, o.gender, o.diabetes, o.passive_smoke_exposure, o.family_respiratory_history,
               o.height, o.weight, o.blood_pressure, o.blood_sugar, o.synthetic,
               i.predicted_class, i.pneumonia_probability, i.confidence, i.uncertainty, i.inference_latency,
               cr.review_status, cr.clinician_reference, cr.clinical_note
        FROM visits v
        LEFT JOIN clinical_observations o ON v.visit_id = o.visit_id
        LEFT JOIN inferences i ON v.visit_id = i.visit_id
        LEFT JOIN clinician_reviews cr ON v.visit_id = cr.visit_id
        WHERE v.visit_id = ?
        """
        row = cursor.execute(query, (visit_id,)).fetchone()
        
        # Also grab any XAI results
        xai_rows = cursor.execute("SELECT method, heatmap_reference FROM xai_results WHERE visit_id = ?", (visit_id,)).fetchall()
        xai_dict = {r['method']: r['heatmap_reference'] for r in xai_rows}
        
        conn.close()
        if not row:
            return None
            
        res = dict(row)
        res['xai_results'] = xai_dict
        return res
