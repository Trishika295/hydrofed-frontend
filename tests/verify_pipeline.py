import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from database.schema import init_database
from database.patient_repository import PatientRepository
from database.visit_repository import VisitRepository
from database.audit_repository import AuditRepository
from cdss.inference_service import InferenceService
from cdss.longitudinal_engine import LongitudinalTrendEngine

def test_full_cdss_pipeline():
    print("--- 1. Testing Database Initialization ---")
    init_database('clinic_local.db')
    p_repo = PatientRepository('clinic_local.db')
    v_repo = VisitRepository('clinic_local.db')
    a_repo = AuditRepository('clinic_local.db')
    
    test_pid = "TEST_PATIENT_2026"
    test_pname = "Alex Rivera"
    
    print("--- 2. Testing Patient Registration & Verification ---")
    if not p_repo.patient_exists(test_pid):
        ok = p_repo.register_patient(test_pid, patient_name=test_pname)
        assert ok, "register_patient failed"
    
    exists = p_repo.patient_exists(test_pid)
    assert exists, "patient_exists verification failed"
    pat = p_repo.get_patient(test_pid)
    print(f"Verified patient: {pat}")
    
    print("--- 3. Testing Visit & Baseline Clinical Observations ---")
    visit_id = f"{test_pid}-V001"
    v_repo.create_visit(visit_id, test_pid, 'Client-01', 'v1.0')
    v_repo.store_clinical_observation(
        visit_id=visit_id,
        age=6.5,
        gender="Male",
        diabetes=0,
        smoke=1,
        family=0,
        height=115.0,
        weight=21.0,
        blood_pressure="105/68",
        blood_sugar=92.0
    )
    a_repo.log_event('PATIENT_REGISTERED', 'Dr. Alisha Nicholls', test_pid, visit_id)
    print("Visit & observations stored.")
    
    print("--- 4. Testing Real InferenceService Execution ---")
    sample_img = "reports/uploaded_xray.jpeg"
    assert os.path.exists(sample_img), f"Sample image {sample_img} does not exist"
    
    inf_service = InferenceService()
    clinical_vector = [6.5 / 80.0, 0.0, 0.0, 1.0, 0.0]
    res = inf_service.run_cdss_diagnostics(
        image_path=sample_img,
        clinical_vector=clinical_vector,
        output_vis_dir='reports/xai_outputs',
        visit_id=visit_id
    )
    print(f"Prediction: {res['predicted_class']}")
    print(f"Pneumonia Prob: {res['pneumonia_probability']:.4f}")
    print(f"Confidence: {res['confidence']:.4f}")
    print(f"Uncertainty: {res['uncertainty']:.4f}")
    print(f"Total Latency: {res['total_cdss_latency']*1000:.1f} ms")
    print(f"Grad-CAM: {res['gradcam_heatmap_path']}")
    print(f"Grad-CAM++: {res['gradcam_plus_heatmap_path']}")
    
    assert os.path.exists(res['gradcam_heatmap_path']), "Grad-CAM file was not generated"
    assert os.path.exists(res['gradcam_plus_heatmap_path']), "Grad-CAM++ file was not generated"
    
    print("--- 5. Storing Inference & XAI in Database ---")
    v_repo.store_inference(
        visit_id=visit_id,
        predicted_class=res['predicted_class'],
        prob=res['pneumonia_probability'],
        confidence=res['confidence'],
        uncertainty=res['uncertainty'],
        latency=res['total_cdss_latency']
    )
    v_repo.store_xai_result(visit_id, 'Grad-CAM', res['gradcam_heatmap_path'])
    v_repo.store_xai_result(visit_id, 'Grad-CAM++', res['gradcam_plus_heatmap_path'])
    a_repo.log_event('INFERENCE_COMPLETED', 'Dr. Alisha Nicholls', test_pid, visit_id)
    
    print("--- 6. Testing Longitudinal Trend Engine ---")
    visits = v_repo.get_visit_history(test_pid)
    engine = LongitudinalTrendEngine()
    trend = engine.analyze_trend(visits)
    print(f"Trend Result: {trend['trend']} - {trend['message']}")
    
    print("--- 7. Testing Audit Repository Retrieval ---")
    logs = a_repo.get_logs(limit=5)
    print(f"Recent audit logs count: {len(logs)}")
    assert len(logs) > 0, "No audit logs found"
    
    print("\nALL BACKEND & INFERENCE INTEGRATION TESTS PASSED SUCCESSFULLY!")

if __name__ == '__main__':
    test_full_cdss_pipeline()
