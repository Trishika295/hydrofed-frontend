import os
import json
import urllib.parse
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import sys

# Add root folder to python path to resolve modules
sys.path.append(os.getcwd())

from database.schema import init_database
from database.patient_repository import PatientRepository
from database.visit_repository import VisitRepository
from database.audit_repository import AuditRepository
from cdss.inference_service import InferenceService
from cdss.longitudinal_engine import LongitudinalTrendEngine

# Global in-memory simulation state
FL_SIMULATION_STATE = {
    'round': 0,
    'consensus_history': [],
    'flow_levels': [0.75, 0.40, 0.60],
    'sec_encrypted_count': 0
}

RUNNER_INSTANCE = None

class DashboardHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # Override to suppress standard HTTP logging to terminal for cleaner outputs
        pass

    def do_GET(self):
        parsed_url = urllib.parse.urlparse(self.path)
        path = parsed_url.path
        query_params = urllib.parse.parse_qs(parsed_url.query)

        # 1. Serve Static Dashboard App
        if path == '/' or path == '/index.html':
            self._serve_file('cdss/static/index.html', 'text/html')
            return
            
        # 2. Serve General Static Files (X-Rays, XAI outputs)
        elif path.startswith('/static/'):
            # Strip '/static/' to get relative path in workspace
            relative_path = path[8:]
            # Clean url encoding
            relative_path = urllib.parse.unquote(relative_path)
            
            # Simple security check to avoid directory traversal
            if '..' in relative_path:
                self.send_error(403, "Access Denied")
                return
                
            if os.path.exists(relative_path):
                content_type = 'image/jpeg' if relative_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png' if relative_path.lower().endswith('.png') else 'application/octet-stream'
                self._serve_file(relative_path, content_type)
            else:
                self.send_error(404, f"File Not Found: {relative_path}")
            return

        # 3. Patient Search API
        elif path == '/api/patients/search':
            query = query_params.get('query', [''])[0]
            repo = PatientRepository()
            
            # Simple filtering
            all_patients = repo.list_all_patients()
            if query:
                filtered = [p for p in all_patients if query.lower() in p['patient_id'].lower()]
            else:
                filtered = all_patients
                
            self._send_json(filtered)
            return

        # 4. Patient Detail API (including visits & longitudinal timeline)
        elif path == '/api/patients/detail':
            patient_id = query_params.get('patient_id', [''])[0]
            p_repo = PatientRepository()
            v_repo = VisitRepository()
            
            patient = p_repo.get_patient(patient_id)
            if not patient:
                self.send_error(404, "Patient Not Found")
                return
                
            visits = v_repo.get_visit_history(patient_id)
            
            # Fallback to check if patient observations are in reports/synthetic_patients_clinical.csv
            age, gender, diabetes, smoke, family = 5.0, "Male", 0, 0, 0
            synthetic_csv = 'reports/synthetic_patients_clinical.csv'
            if os.path.exists(synthetic_csv):
                import pandas as pd
                df = pd.read_csv(synthetic_csv)
                match = df[df['patient_id'] == patient_id]
                if not match.empty:
                    row = match.iloc[0]
                    age = float(row['age'])
                    gender = row['gender']
                    diabetes = int(row['diabetes'])
                    smoke = int(row['passive_smoke_exposure'])
                    family = int(row['family_respiratory_history'])

            res = {
                'patient_id': patient_id,
                'created_at': patient['created_at'],
                'status': patient['status'],
                'age': age,
                'gender': gender,
                'diabetes': diabetes,
                'passive_smoke_exposure': smoke,
                'family_respiratory_history': family,
                'visits': visits
            }
            self._send_json(res)
            return

        # 5. Security Audit Log API
        elif path == '/api/audit/logs':
            repo = AuditRepository()
            logs = repo.get_logs(limit=100)
            self._send_json(logs)
            return

        else:
            self.send_error(404, "API endpoint not found")

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode('utf-8'))

        path = self.path

        # 1. Patient Registration API
        if path == '/api/patients/register':
            patient_id = data.get('patient_id')
            age = float(data.get('age', 5.0))
            gender = data.get('gender', 'Male')
            diabetes = int(data.get('diabetes', 0))
            smoke = int(data.get('passive_smoke_exposure', 0))
            family = int(data.get('family_respiratory_history', 0))

            p_repo = PatientRepository()
            v_repo = VisitRepository()
            a_repo = AuditRepository()

            if p_repo.patient_exists(patient_id):
                self._send_json({'success': False, 'message': 'Patient ID already registered'})
                return

            # Insert patient
            success = p_repo.register_patient(patient_id)
            if success:
                # Create a baseline visit representation and observation
                visit_id = f"{patient_id}-V001"
                v_repo.create_visit(visit_id, patient_id, 'Client-01', 'v1.0')
                v_repo.store_clinical_observation(visit_id, age, gender, diabetes, smoke, family)
                
                # Write to audit trail
                a_repo.log_event('PATIENT_REGISTERED', 'CLINICIAN', patient_id, visit_id)
                self._send_json({'success': True, 'patient_id': patient_id})
            else:
                self._send_json({'success': False, 'message': 'Database insert failure'})
            return

        # 2. CDSS Diagnostics Runner API
        elif path == '/api/inferences/run':
            filepath = data.get('filepath')
            image_base64 = data.get('image_base64')
            patient_id = data.get('patient_id')

            if image_base64:
                import base64
                if ',' in image_base64:
                    image_base64 = image_base64.split(',')[1]
                img_bytes = base64.b64decode(image_base64)
                os.makedirs('reports', exist_ok=True)
                filepath = 'reports/uploaded_xray.jpeg'
                with open(filepath, 'wb') as f:
                    f.write(img_bytes)

            # Look up clinical metadata in synthetic record
            age, gender, diabetes, smoke, family = 5.0, "Male", 0, 0, 0
            synthetic_csv = 'reports/synthetic_patients_clinical.csv'
            if os.path.exists(synthetic_csv):
                import pandas as pd
                df = pd.read_csv(synthetic_csv)
                match = df[df['patient_id'] == patient_id]
                if not match.empty:
                    row = match.iloc[0]
                    age = float(row['age'])
                    gender = row['gender']
                    diabetes = int(row['diabetes'])
                    smoke = int(row['passive_smoke_exposure'])
                    family = int(row['family_respiratory_history'])

            # Normalize clinical variables for model input (matching dataset_loader.py)
            age_norm = age / 80.0
            gender_val = 1.0 if gender == 'Female' else 0.0
            clinical_vector = [age_norm, gender_val, float(diabetes), float(smoke), float(family)]

            # Instantiate inference service
            # (Note: we don't load external checkpoints here, so it uses initialized weights for demonstration)
            inf_service = InferenceService()
            visit_id = f"{patient_id}-V{int(time.time()) % 1000:03d}"
            
            print(f"Executing diagnostic pass for {patient_id} on {filepath}...")
            # Run diagnostics pipeline
            res = inf_service.run_cdss_diagnostics(
                image_path=filepath,
                clinical_vector=clinical_vector,
                output_vis_dir='reports/xai_outputs',
                visit_id=visit_id
            )

            # Save preprocessed image visual preview
            import cv2
            import numpy as np
            try:
                raw_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if raw_img is not None:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    clahe_img = clahe.apply(raw_img)
                    h, w = clahe_img.shape[:2]
                    scale = min(224 / w, 224 / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = cv2.resize(clahe_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    padded = np.zeros((224, 224), dtype=np.uint8)
                    x_offset = (224 - new_w) // 2
                    y_offset = (224 - new_h) // 2
                    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
                    prep_vis_path = f"reports/xai_outputs/{visit_id}_preprocessed.png"
                    cv2.imwrite(prep_vis_path, padded)
                else:
                    prep_vis_path = filepath
            except Exception as e:
                print(f"Warning: Failed to save preprocessed image vis: {e}")
                prep_vis_path = filepath

            # Store in DB
            v_repo = VisitRepository()
            a_repo = AuditRepository()
            
            v_repo.create_visit(visit_id, patient_id, 'Client-01', 'v1.0')
            v_repo.store_clinical_observation(visit_id, age, gender, diabetes, smoke, family)
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

            a_repo.log_event('INFERENCE_COMPLETED', 'CLINICIAN', patient_id, visit_id)

            # Return endpoints relative paths
            response_payload = {
                'visit_id': visit_id,
                'predicted_class': res['predicted_class'],
                'pneumonia_probability': res['pneumonia_probability'],
                'confidence': res['confidence'],
                'uncertainty': res['uncertainty'],
                'total_cdss_latency': res['total_cdss_latency'],
                'gradcam_url': '/static/' + res['gradcam_heatmap_path'].replace('\\', '/'),
                'gradcam_plus_url': '/static/' + res['gradcam_plus_heatmap_path'].replace('\\', '/'),
                'preprocessed_url': '/static/' + prep_vis_path.replace('\\', '/'),
                'clinical_data': {
                    'age': age,
                    'gender': gender,
                    'diabetes': diabetes,
                    'passive_smoke_exposure': smoke,
                    'family_respiratory_history': family
                }
            }
            self._send_json(response_payload)
            return

        # 3. Clinician Review Submit API
        elif path == '/api/reviews/store':
            clinician_ref = data.get('clinician_reference')
            review_status = data.get('review_status')
            clinical_note = data.get('clinical_note')
            
            v_repo = VisitRepository()
            a_repo = AuditRepository()
            
            # Map review to the latest visit
            # For demo, grab the very last audit entry or latest visit ID in visits
            conn = get_db_connection('clinic_local.db')
            last_visit = conn.execute("SELECT visit_id, patient_id FROM visits ORDER BY visit_timestamp DESC LIMIT 1").fetchone()
            conn.close()

            if last_visit:
                visit_id = last_visit['visit_id']
                patient_id = last_visit['patient_id']
                
                v_repo.store_clinician_review(visit_id, review_status, clinician_ref, clinical_note)
                a_repo.log_event('CLINICIAN_REVIEWED', f"CLINICIAN: {clinician_ref}", patient_id, visit_id)
                self._send_json({'success': True, 'visit_id': visit_id})
            else:
                self._send_json({'success': False, 'message': 'No active visit found'})
            return

        # 4. Simulate Gossip FL round API
        elif path == '/api/federated/simulate_round':
            global RUNNER_INSTANCE
            if RUNNER_INSTANCE is None:
                from experiments.run_experiment import UnifiedExperimentRunner
                RUNNER_INSTANCE = UnifiedExperimentRunner()
                
            # Run one actual communication round of HydroFed decentralized training
            test_metrics, consensus_errors = RUNNER_INSTANCE.run_decentralized_hydrofed(rounds=1)
            err = float(consensus_errors[-1]) if len(consensus_errors) > 0 else 0.05
            
            FL_SIMULATION_STATE['round'] += 1
            r = FL_SIMULATION_STATE['round']
            FL_SIMULATION_STATE['consensus_history'].append(err)
            
            # Simulate water levels convergence: they get closer to average (e.g. 0.58)
            avg = 0.58
            for i in range(len(FL_SIMULATION_STATE['flow_levels'])):
                curr = FL_SIMULATION_STATE['flow_levels'][i]
                FL_SIMULATION_STATE['flow_levels'][i] = curr + 0.5 * (avg - curr) + (np.random.rand() - 0.5) * 0.02

            # Simulate encryption overhead (60 updates encrypted via AES-256-GCM)
            FL_SIMULATION_STATE['sec_encrypted_count'] += 60
            
            # Log to SQLite audit log
            a_repo = AuditRepository()
            a_repo.log_event('FL_UPDATE_SENT', 'FL_CLIENT', None, None)
            a_repo.log_event('MODEL_UPDATED', 'HYDROFED_ENGINE', None, None)
            
            res = {
                'round': r,
                'consensus_history': FL_SIMULATION_STATE['consensus_history'],
                'flow_levels': FL_SIMULATION_STATE['flow_levels']
            }
            self._send_json(res)
            return

        else:
            self.send_error(404, "API endpoint not found")

    def _serve_file(self, filepath, content_type):
        try:
            with open(filepath, 'rb') as f:
                content = f.read()
            self.send_response(200)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error serving file: {e}")

    def _send_json(self, data):
        content = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(content))
        self.end_headers()
        self.wfile.write(content)

import numpy as np

def run_server(port=8080):
    import subprocess
    import sys
    init_database()
    # Pre-register some mock patients in database for easy searching
    p_repo = PatientRepository()
    if len(p_repo.list_all_patients()) == 0:
        print("Pre-registering mock patients for clinical demonstration...")
        p_repo.register_patient('person100')
        p_repo.register_patient('person101')
        p_repo.register_patient('normal_0001')
        
    print(f"Launching Streamlit Portal on port {port}...")
    cmd = [sys.executable, "-m", "streamlit", "run", "cdss/app.py", "--server.port", str(port)]
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nStopping portal...")

if __name__ == '__main__':
    run_server(8080)

