import unittest
import os
from database.schema import init_database
from database.patient_repository import PatientRepository
from database.visit_repository import VisitRepository
from database.audit_repository import AuditRepository

class TestClinicDatabase(unittest.TestCase):
    def setUp(self):
        self.db_path = 'test_clinic_temp.db'
        init_database(self.db_path)
        self.p_repo = PatientRepository(self.db_path)
        self.v_repo = VisitRepository(self.db_path)
        self.a_repo = AuditRepository(self.db_path)

    def tearDown(self):
        # Clean up database file after test run
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except PermissionError:
                pass

    def test_database_e2e_workflow(self):
        pid = "P-TEST-9999"
        
        # 1. Register Patient
        registered = self.p_repo.register_patient(pid)
        self.assertTrue(registered)
        self.assertTrue(self.p_repo.patient_exists(pid))
        
        # 2. Add Visit
        vid = f"{pid}-V01"
        visit_created = self.v_repo.create_visit(vid, pid, "Client-01", "v1.0")
        self.assertTrue(visit_created)
        
        # 3. Store Clinical Observations
        obs_stored = self.v_repo.store_clinical_observation(
            visit_id=vid, age=6.2, gender="Female", diabetes=1, smoke=0, family=1
        )
        self.assertTrue(obs_stored)
        
        # 4. Store Model Inference
        inf_stored = self.v_repo.store_inference(
            visit_id=vid, predicted_class=1, prob=0.82, confidence=0.88, uncertainty=0.06, latency=0.125
        )
        self.assertTrue(inf_stored)
        
        # 5. Store Clinician Review
        rev_stored = self.v_repo.store_clinician_review(
            visit_id=vid, review_status="CONFIRMED", clinician_ref="Dr. Adams", clinical_note="Infiltrates verified in left lung lobe"
        )
        self.assertTrue(rev_stored)
        
        # 6. Retrieve Patient visit timeline
        history = self.v_repo.get_visit_history(pid)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['visit_id'], vid)
        self.assertEqual(history[0]['age'], 6.2)
        self.assertEqual(history[0]['predicted_class'], 1)
        self.assertEqual(history[0]['review_status'], "CONFIRMED")
        
        # 7. Audit log event check
        logged = self.a_repo.log_event('TEST_EVENT', 'TEST_ACTOR', pid, vid)
        self.assertTrue(logged)
        
        logs = self.a_repo.get_logs(limit=10)
        self.assertTrue(len(logs) > 0)
        self.assertEqual(logs[0]['event_type'], 'TEST_EVENT')
        self.assertEqual(logs[0]['patient_id'], pid)

if __name__ == '__main__':
    unittest.main()
