import os
import sys
import time
import base64
import json
import psutil
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

# Ensure repository root is on Python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from database.schema import init_database, get_db_connection
from database.patient_repository import PatientRepository
from database.visit_repository import VisitRepository
from database.audit_repository import AuditRepository
from cdss.inference_service import InferenceService
from cdss.longitudinal_engine import LongitudinalTrendEngine

# -----------------------------------------------------------------------------
# App Configuration & Initialization
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HydroFed-ICAF | Clinical Decision Support System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Database
init_database('clinic_local.db')
p_repo = PatientRepository('clinic_local.db')
if len(p_repo.list_all_patients()) == 0:
    p_repo.register_patient('person100', patient_name='Liam Miller')
    p_repo.register_patient('person101', patient_name='Sophia Chen')
    p_repo.register_patient('normal_0001', patient_name='Noah Williams')

# Cache heavyweight inference service
@st.cache_resource
def get_inference_service():
    return InferenceService()

# -----------------------------------------------------------------------------
# Session State Context Initialization (Full Compliance)
# -----------------------------------------------------------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'clinician_name' not in st.session_state:
    st.session_state.clinician_name = "Dr. Alisha Nicholls"
if 'clinician_role' not in st.session_state:
    st.session_state.clinician_role = "Pediatric Pulmonologist"

# Mandatory session state keys
if 'selected_patient_id' not in st.session_state:
    st.session_state.selected_patient_id = ""
if 'active_patient_id' not in st.session_state:
    st.session_state.active_patient_id = ""
if 'selected_patient' not in st.session_state:
    st.session_state.selected_patient = None
if 'active_patient_data' not in st.session_state:
    st.session_state.active_patient_data = None
if 'selected_xray' not in st.session_state:
    st.session_state.selected_xray = None
if 'current_evaluation' not in st.session_state:
    st.session_state.current_evaluation = None
if 'current_cdss_result' not in st.session_state:
    st.session_state.current_cdss_result = None
if 'xai_outputs' not in st.session_state:
    st.session_state.xai_outputs = None
if 'diagnostic_results' not in st.session_state:
    st.session_state.diagnostic_results = None
if 'current_nav' not in st.session_state:
    st.session_state.current_nav = "Home"
if 'reg_verified_id' not in st.session_state:
    st.session_state.reg_verified_id = None

def sync_patient(pid, clinical_data=None):
    """Synchronizes selected_patient_id, active_patient_id, selected_patient, and active_patient_data."""
    st.session_state.selected_patient_id = pid
    st.session_state.active_patient_id = pid
    if pid:
        p_repo_inst = PatientRepository('clinic_local.db')
        st.session_state.selected_patient = p_repo_inst.get_patient(pid)
        if clinical_data is not None:
            st.session_state.active_patient_data = clinical_data
        else:
            v_repo_inst = VisitRepository('clinic_local.db')
            hist = v_repo_inst.get_visit_history(pid)
            if hist:
                latest = hist[-1]
                st.session_state.active_patient_data = {
                    'age': float(latest.get('age', 5.0) or 5.0),
                    'gender': latest.get('gender', 'Male') or 'Male',
                    'diabetes': int(latest.get('diabetes', 0) or 0),
                    'passive_smoke_exposure': int(latest.get('passive_smoke_exposure', 0) or 0),
                    'family_respiratory_history': int(latest.get('family_respiratory_history', 0) or 0),
                    'height': float(latest.get('height', 110.0) or 110.0),
                    'weight': float(latest.get('weight', 19.5) or 19.5),
                    'blood_pressure': latest.get('blood_pressure', '110/70') or '110/70',
                    'blood_sugar': float(latest.get('blood_sugar', 95.0) or 95.0)
                }
            else:
                st.session_state.active_patient_data = {
                    'age': 5.0, 'gender': 'Male', 'diabetes': 0, 'passive_smoke_exposure': 0,
                    'family_respiratory_history': 0, 'height': 110.0, 'weight': 19.5,
                    'blood_pressure': '110/70', 'blood_sugar': 95.0
                }
    else:
        st.session_state.selected_patient = None
        st.session_state.active_patient_data = None

# Ensure two-way synchronization
if st.session_state.selected_patient_id and not st.session_state.active_patient_id:
    st.session_state.active_patient_id = st.session_state.selected_patient_id
elif st.session_state.active_patient_id and not st.session_state.selected_patient_id:
    st.session_state.selected_patient_id = st.session_state.active_patient_id

# -----------------------------------------------------------------------------
# Custom Clinical UI/UX Styling (Medical Dashboard Inspiration)
# -----------------------------------------------------------------------------
def inject_medical_theme():
    st.markdown("""
    <style>
        /* Modern Font Imports & Base Reset */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
            color: #1e293b;
        }

        /* App Background */
        .stApp {
            background-color: #f3f6fc !important;
        }

        /* Sidebar Medical Theme */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #1d4ed8 0%, #2563eb 100%) !important;
            border-right: none !important;
            box-shadow: 4px 0 20px rgba(37, 99, 235, 0.1) !important;
        }
        
        [data-testid="stSidebar"] * {
            color: #ffffff !important;
        }

        /* Sidebar Logo Container */
        .sidebar-brand-container {
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 16px 12px 24px 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.15);
            margin-bottom: 18px;
        }

        .brand-icon-box {
            width: 44px;
            height: 44px;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(8px);
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        }

        .brand-text-title {
            font-size: 1.15rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            color: #ffffff;
            margin: 0;
            line-height: 1.2;
        }

        .brand-text-subtitle {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: #bfdbfe;
            margin: 0;
            font-weight: 600;
        }

        /* Sidebar Navigation Radio Styling */
        [data-testid="stSidebar"] .stRadio > label {
            display: none !important;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] {
            gap: 6px;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            margin-bottom: 6px !important;
            cursor: pointer !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
            background: rgba(255, 255, 255, 0.2) !important;
            transform: translateX(4px);
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"],
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) {
            background: #ffffff !important;
            border-color: #ffffff !important;
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.12) !important;
        }

        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label[data-checked="true"] *,
        [data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:has(input:checked) * {
            color: #1d4ed8 !important;
            font-weight: 700 !important;
        }

        /* Top Header Container */
        .medical-top-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #ffffff;
            padding: 14px 24px;
            border-radius: 18px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.03);
            margin-bottom: 24px;
        }

        .header-search-bar {
            display: flex;
            align-items: center;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 8px 16px;
            width: 380px;
            max-width: 100%;
            color: #64748b;
            font-size: 0.88rem;
        }

        .header-search-bar input {
            border: none;
            background: transparent;
            outline: none;
            margin-left: 8px;
            width: 100%;
            color: #1e293b;
        }

        .header-profile-box {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .profile-avatar-circle {
            width: 44px;
            height: 44px;
            background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            border: 2px solid #3b82f6;
        }

        .profile-info-text {
            display: flex;
            flex-direction: column;
        }

        .profile-name {
            font-weight: 700;
            color: #0f172a;
            font-size: 0.95rem;
            line-height: 1.2;
        }

        .profile-title {
            color: #64748b;
            font-size: 0.78rem;
            font-weight: 500;
        }

        /* Medical Welcome Banner */
        .welcome-banner-card {
            background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
            border-radius: 20px;
            padding: 30px;
            color: #ffffff;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.25);
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: relative;
            overflow: hidden;
        }

        .welcome-banner-card::before {
            content: "";
            position: absolute;
            top: -40px;
            right: -40px;
            width: 200px;
            height: 200px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 50%;
        }

        .banner-badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(8px);
            padding: 5px 14px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 600;
            margin-bottom: 12px;
            letter-spacing: 0.3px;
        }

        .banner-heading {
            font-size: 1.85rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            color: #ffffff !important;
            letter-spacing: -0.5px;
        }

        .banner-desc {
            font-size: 0.95rem;
            color: #dbeafe;
            margin: 0;
            max-width: 650px;
            line-height: 1.5;
        }

        /* Patient Header Banner */
        .active-patient-card {
            background: #ffffff;
            border-radius: 16px;
            border-left: 6px solid #2563eb;
            border: 1px solid #e2e8f0;
            padding: 16px 22px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
        }

        .patient-tag {
            display: inline-block;
            background: #eff6ff;
            color: #2563eb;
            font-weight: 700;
            padding: 4px 10px;
            border-radius: 8px;
            font-size: 0.8rem;
            margin-right: 8px;
        }

        /* Generic Medical Card */
        .med-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 22px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 6px 20px rgba(15, 23, 42, 0.03);
            margin-bottom: 20px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }

        .med-card:hover {
            box-shadow: 0 10px 25px rgba(15, 23, 42, 0.06);
        }

        .card-header-label {
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 700;
            color: #64748b;
            margin-bottom: 6px;
        }

        .card-stat-value {
            font-size: 1.75rem;
            font-weight: 800;
            color: #0f172a;
            margin: 0;
            line-height: 1.2;
        }

        .card-subtext {
            font-size: 0.82rem;
            color: #94a3b8;
            margin-top: 6px;
        }

        /* Badges */
        .badge-status {
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.3px;
        }
        .badge-green { background: #ecfdf5; color: #059669; border: 1px solid #a7f3d0; }
        .badge-red { background: #fef2f2; color: #dc2626; border: 1px solid #fecaca; }
        .badge-blue { background: #eff6ff; color: #2563eb; border: 1px solid #bfdbfe; }
        .badge-amber { background: #fffbeb; color: #d97706; border: 1px solid #fde68a; }

        /* Buttons */
        .stButton button {
            border-radius: 12px !important;
            font-weight: 600 !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
            border: 1px solid transparent !important;
        }

        .stButton button:hover {
            transform: translateY(-1px) !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.15) !important;
        }

        /* Dataframes & Tables */
        [data-testid="stDataFrame"] {
            border-radius: 14px !important;
            overflow: hidden !important;
            border: 1px solid #e2e8f0 !important;
        }

        /* Input Controls */
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            border-radius: 10px !important;
            border: 1px solid #cbd5e1 !important;
            background-color: #ffffff !important;
        }

        /* Responsive Layout Overrides */
        @media (max-width: 1024px) {
            .welcome-banner-card { padding: 22px; }
            .banner-heading { font-size: 1.5rem; }
            .header-search-bar { width: 240px; }
        }

        @media (max-width: 768px) {
            .medical-top-header { flex-direction: column; gap: 12px; align-items: flex-start; }
            .header-search-bar { width: 100%; }
            .welcome-banner-card { flex-direction: column; align-items: flex-start; }
            .active-patient-card { flex-direction: column; align-items: flex-start; gap: 10px; }
            [data-testid="column"] {
                min-width: 100% !important;
                flex: 1 1 100% !important;
                margin-bottom: 12px !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# Helper for clean light-themed Matplotlib charts
def setup_light_matplotlib():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Inter', 'DejaVu Sans', 'Arial']
    plt.rcParams['text.color'] = '#1e293b'
    plt.rcParams['axes.labelcolor'] = '#475569'
    plt.rcParams['xtick.color'] = '#64748b'
    plt.rcParams['ytick.color'] = '#64748b'
    plt.rcParams['axes.edgecolor'] = '#e2e8f0'
    plt.rcParams['grid.color'] = '#f1f5f9'
    plt.rcParams['figure.facecolor'] = '#ffffff'
    plt.rcParams['axes.facecolor'] = '#ffffff'

# -----------------------------------------------------------------------------
# Component: Top Header
# -----------------------------------------------------------------------------
def render_top_header():
    clinician = st.session_state.clinician_name
    role = st.session_state.clinician_role
    st.markdown(f"""
    <div class="medical-top-header">
        <div class="header-search-bar">
            <span>🔍</span>
            <input type="text" placeholder="Search patient ID, visits, radiological records..." readonly/>
        </div>
        <div class="header-profile-box">
            <span class="badge-status badge-blue">🟢 Edge AI Node 01 | Online</span>
            <div class="profile-avatar-circle">🩺</div>
            <div class="profile-info-text">
                <span class="profile-name">{clinician}</span>
                <span class="profile-title">{role}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Component: Active Patient Banner
# -----------------------------------------------------------------------------
def render_active_patient_banner():
    active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
    if active_pid:
        p_info = PatientRepository('clinic_local.db').get_patient(active_pid)
        pname = p_info.get('patient_name', active_pid) if p_info else active_pid
        
        data = st.session_state.active_patient_data or {}
        age_str = f"{data.get('age', 'N/A')} yrs" if 'age' in data else "N/A"
        gender_str = data.get('gender', 'N/A')
        
        st.markdown(f"""
        <div class="active-patient-card">
            <div>
                <span class="patient-tag">SELECTED PATIENT CONTEXT</span>
                <strong style="font-size: 1.15rem; color: #0f172a; margin-right: 12px;">{pname}</strong>
                <span style="color: #64748b; font-size: 0.9rem; margin-right: 14px;">ID: <strong>{active_pid}</strong></span>
                <span style="color: #64748b; font-size: 0.9rem; margin-right: 14px;">Age: <strong>{age_str}</strong></span>
                <span style="color: #64748b; font-size: 0.9rem;">Gender: <strong>{gender_str}</strong></span>
            </div>
            <div>
                <span class="badge-status badge-green">Ready for CDSS Analysis</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="active-patient-card" style="border-left-color: #f59e0b;">
            <div>
                <span class="badge-status badge-amber">No Active Patient</span>
                <span style="color: #475569; font-size: 0.9rem; margin-left: 8px;">
                    Please select or register a patient profile to enable diagnostics.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Authentication View
# -----------------------------------------------------------------------------
def render_login_view():
    st.markdown("""
    <div style="max-width: 480px; margin: 60px auto 30px auto; text-align: center;">
        <div style="width: 70px; height: 70px; background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%); border-radius: 20px; display: inline-flex; align-items: center; justify-content: center; font-size: 34px; box-shadow: 0 10px 25px rgba(37, 99, 235, 0.3); margin-bottom: 20px;">🩺</div>
        <h1 style="font-size: 2rem; font-weight: 800; color: #0f172a; margin-bottom: 6px;">HYDROFED-ICAF</h1>
        <p style="color: #64748b; font-size: 1rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 24px;">Clinical Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        col_pad1, col_center, col_pad2 = st.columns([1, 1.4, 1])
        with col_center:
            st.markdown("""
            <div class="med-card" style="padding: 36px 32px; border-radius: 24px; box-shadow: 0 15px 35px rgba(15, 23, 42, 0.06);">
                <div class="card-header-label" style="text-align: center; margin-bottom: 18px;">Secure Clinical Workspace</div>
            """, unsafe_allow_html=True)
            
            with st.form("clinical_login_form"):
                username = st.text_input("Clinician Username", placeholder="e.g. doctor or clinic_admin", value="doctor")
                password = st.text_input("Password", type="password", value="hydrofed2026")
                submitted = st.form_submit_button("Login to Workspace", width="stretch")
                
                if submitted:
                    if username.strip() and password.strip():
                        st.session_state.authenticated = True
                        st.session_state.clinician_name = f"Dr. {username.capitalize()}" if not username.startswith("Dr.") else username
                        AuditRepository('clinic_local.db').log_event('CLINICIAN_LOGIN', username)
                        st.success("Authentication successful. Redirecting to workspace...")
                        time.sleep(0.4)
                        st.rerun()
                    else:
                        st.error("Please provide valid clinician credentials.")

            st.markdown("""
                <div style="text-align: center; margin-top: 18px; font-size: 0.8rem; color: #94a3b8;">
                    Demo Credentials: <code>doctor</code> / <code>hydrofed2026</code>
                </div>
            </div>
            <div style="text-align: center; margin-top: 24px; color: #64748b; font-size: 0.82rem;">
                Research Prototype — AI-assisted clinical decision support.
            </div>
            """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAIN APPLICATION WORKFLOW
# -----------------------------------------------------------------------------
inject_medical_theme()

if not st.session_state.authenticated:
    render_login_view()
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR (EXACTLY 10 SECTIONS + LOGOUT)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand-container">
        <div class="brand-icon-box">🩺</div>
        <div>
            <div class="brand-text-title">HYDROFED-ICAF</div>
            <div class="brand-text-subtitle">Clinical Decision Support</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    NAVIGATION_ITEMS = [
        "Home",
        "Patient Registration",
        "Patient Search and Selection",
        "X-Ray Analysis",
        "Active Clinical Inputs",
        "CDSS Classifier",
        "Pathology Salience (XAI)",
        "Longitudinal Trend",
        "Edge Performance Metrics",
        "System Security Audit Log"
    ]

    selected_nav = st.radio(
        "Navigation",
        NAVIGATION_ITEMS,
        index=NAVIGATION_ITEMS.index(st.session_state.current_nav) if st.session_state.current_nav in NAVIGATION_ITEMS else 0,
        key="sidebar_radio"
    )
    st.session_state.current_nav = selected_nav

    st.markdown("<br><hr style='border-color: rgba(255, 255, 255, 0.15); margin: 15px 0;'/>", unsafe_allow_html=True)

    # Active patient indicator in sidebar
    active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
    if active_pid:
        p_row = PatientRepository('clinic_local.db').get_patient(active_pid)
        disp_name = p_row.get('patient_name', active_pid) if p_row else active_pid
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.15); border-radius: 12px; padding: 12px; margin-bottom: 12px;">
            <div style="font-size: 0.72rem; color: #bfdbfe; font-weight: 700; text-transform: uppercase;">ACTIVE PATIENT</div>
            <div style="font-size: 0.95rem; font-weight: 700; color: #ffffff;">{disp_name}</div>
            <div style="font-size: 0.78rem; color: #e0e7ff;">ID: {active_pid}</div>
        </div>
        """, unsafe_allow_html=True)

    # Logout Button
    if st.button("🚪 Logout", width="stretch"):
        AuditRepository('clinic_local.db').log_event('CLINICIAN_LOGOUT', st.session_state.clinician_name)
        st.session_state.authenticated = False
        st.session_state.selected_patient_id = ""
        st.session_state.active_patient_id = ""
        st.session_state.selected_patient = None
        st.session_state.active_patient_data = None
        st.session_state.selected_xray = None
        st.session_state.current_evaluation = None
        st.session_state.current_cdss_result = None
        st.session_state.xai_outputs = None
        st.session_state.diagnostic_results = None
        st.session_state.reg_verified_id = None
        st.rerun()

# -----------------------------------------------------------------------------
# PAGE 1: HOME
# -----------------------------------------------------------------------------
if selected_nav == "Home":
    render_top_header()
    
    clinician = st.session_state.clinician_name
    curr_date = time.strftime("%B %d, %Y • %I:%M %p")
    
    st.markdown(f"""
    <div class="welcome-banner-card">
        <div>
            <div class="banner-badge">📅 {curr_date}</div>
            <h1 class="banner-heading">Good Day, {clinician}!</h1>
            <p class="banner-desc">
                Welcome to the HydroFed-ICAF Clinical Workspace — A Bio-Inspired Multimodal Decentralized Edge-AI Clinical Decision Support System.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Real Database Metrics Query
    conn = get_db_connection('clinic_local.db')
    registered_patients_count = len(PatientRepository('clinic_local.db').list_all_patients())
    total_evaluations = conn.execute("SELECT COUNT(*) FROM inferences").fetchone()[0]
    total_reviews = conn.execute("SELECT COUNT(*) FROM clinician_reviews").fetchone()[0]
    conn.close()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">REGISTERED PATIENTS</div>
            <div class="card-stat-value">{registered_patients_count}</div>
            <div class="card-subtext">Active local clinic records</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">X-RAY EVALUATIONS</div>
            <div class="card-stat-value" style="color: #2563eb;">{total_evaluations}</div>
            <div class="card-subtext">Multi-pass MC evaluations run</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">CLINICIAN REVIEWS</div>
            <div class="card-stat-value" style="color: #059669;">{total_reviews}</div>
            <div class="card-subtext">Signed clinical records</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="med-card">
            <div class="card-header-label">SYSTEM STATUS</div>
            <div class="card-stat-value" style="color: #059669; font-size: 1.45rem;">NOMINAL</div>
            <div class="card-subtext">Edge CPU | AES-256-GCM</div>
        </div>
        """, unsafe_allow_html=True)

    # Recent Evaluations Table
    st.markdown("### Recent CDSS Evaluations")
    conn = get_db_connection('clinic_local.db')
    eval_rows = conn.execute("""
        SELECT v.patient_id as "Patient ID",
               COALESCE(p.patient_name, v.patient_id) as "Patient Name",
               v.visit_timestamp as "Evaluation Date",
               CASE WHEN i.predicted_class = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as "Prediction",
               ROUND(i.pneumonia_probability * 100, 1) || '%' as "Probability",
               ROUND(i.uncertainty, 4) as "Uncertainty"
        FROM visits v
        JOIN inferences i ON v.visit_id = i.visit_id
        LEFT JOIN patients p ON v.patient_id = p.patient_id
        ORDER BY v.visit_timestamp DESC LIMIT 8
    """).fetchall()
    conn.close()

    if eval_rows:
        df_recent = pd.DataFrame([dict(r) for r in eval_rows])
        st.dataframe(df_recent, width="stretch")
    else:
        st.markdown("""
        <div class="med-card" style="text-align: center; padding: 30px; color: #64748b;">
            <span style="font-size: 2rem;">🩻</span><br>
            <strong>No evaluations recorded yet.</strong><br>
            Select a patient and run an X-Ray diagnostic analysis to record evaluations in the local database.
        </div>
        """, unsafe_allow_html=True)

    # Medical Workflow Quick Navigation
    st.markdown("### CDSS Clinical Workflow")
    w1, w2, w3, w4 = st.columns(4)
    with w1:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <strong>1. Patient Selection</strong>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 6px;">Lookup or register patient demographic records securely on local storage.</p>
        </div>
        """, unsafe_allow_html=True)
    with w2:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <strong>2. X-Ray Analysis</strong>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 6px;">Upload radiological scans or use verified hospital test cases for evaluation.</p>
        </div>
        """, unsafe_allow_html=True)
    with w3:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <strong>3. Multimodal CDSS</strong>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 6px;">DenseNet-151 + IIFR frequency fusion & AICA cross-attention with MC Dropout.</p>
        </div>
        """, unsafe_allow_html=True)
    with w4:
        st.markdown("""
        <div class="med-card" style="height: 100%;">
            <strong>4. XAI & Longitudinal</strong>
            <p style="color: #64748b; font-size: 0.85rem; margin-top: 6px;">Examine Grad-CAM/++ overlays and track temporal disease progression indices.</p>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PAGE 2: PATIENT REGISTRATION
# -----------------------------------------------------------------------------
elif selected_nav == "Patient Registration":
    render_top_header()
    
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Patient Registration</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Register a new patient into the local clinic database with baseline clinical demographics.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("patient_registration_form"):
        st.markdown("#### Patient Information")
        col1, col2 = st.columns(2)
        with col1:
            patient_id = st.text_input("Patient ID *", placeholder="e.g. clinic_patient_800 or P-2026-042")
            patient_name = st.text_input("Patient Name *", placeholder="e.g. Liam Miller")
        with col2:
            age = st.number_input("Age (Years) *", min_value=0.1, max_value=120.0, value=5.0, step=0.1)
            gender = st.selectbox("Gender *", ["Male", "Female"])

        st.markdown("<hr style='border-color: #e2e8f0; margin: 15px 0;'/>", unsafe_allow_html=True)
        st.markdown("#### Baseline Clinical Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            height = st.number_input("Height (cm)", min_value=0.0, max_value=250.0, value=110.0, step=0.5)
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=200.0, value=19.5, step=0.5)
        with c2:
            blood_pressure = st.text_input("Blood Pressure (mmHg)", placeholder="e.g. 110/70", value="110/70")
            blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=0.0, max_value=500.0, value=95.0, step=1.0)
        with c3:
            diabetes = st.selectbox("Diabetes Mellitus", ["Absent", "Present"])
            smoke = st.selectbox("Passive Smoke Exposure", ["No", "Yes"])
            family = st.selectbox("Family Respiratory History", ["No", "Yes"])

        submitted = st.form_submit_button("Register Patient", width="stretch")
        
        if submitted:
            # Field Validations
            clean_id = patient_id.strip()
            clean_name = patient_name.strip()
            
            if not clean_id:
                st.error("Patient ID is required.")
            elif not clean_name:
                st.error("Patient Name is required.")
            elif age <= 0:
                st.error("Please enter a valid age greater than 0.")
            else:
                p_repo_inst = PatientRepository('clinic_local.db')
                v_repo_inst = VisitRepository('clinic_local.db')
                a_repo_inst = AuditRepository('clinic_local.db')

                if p_repo_inst.patient_exists(clean_id):
                    st.error(f"Patient ID '{clean_id}' already exists in the database. Please use a unique ID.")
                else:
                    success = p_repo_inst.register_patient(clean_id, patient_name=clean_name)
                    
                    if success:
                        visit_id = f"{clean_id}-V001"
                        v_repo_inst.create_visit(visit_id, clean_id, 'Client-01', 'v1.0')
                        v_repo_inst.store_clinical_observation(
                            visit_id=visit_id,
                            age=age,
                            gender=gender,
                            diabetes=1 if diabetes == "Present" else 0,
                            smoke=1 if smoke == "Yes" else 0,
                            family=1 if family == "Yes" else 0,
                            height=height,
                            weight=weight,
                            blood_pressure=blood_pressure,
                            blood_sugar=blood_sugar
                        )
                        a_repo_inst.log_event('PATIENT_REGISTERED', st.session_state.clinician_name, clean_id, visit_id)

                        # VERIFY DATABASE SAVE
                        if p_repo_inst.patient_exists(clean_id):
                            patient_record = {
                                'age': age,
                                'gender': gender,
                                'diabetes': 1 if diabetes == "Present" else 0,
                                'passive_smoke_exposure': 1 if smoke == "Yes" else 0,
                                'family_respiratory_history': 1 if family == "Yes" else 0,
                                'height': height,
                                'weight': weight,
                                'blood_pressure': blood_pressure,
                                'blood_sugar': blood_sugar
                            }
                            st.session_state.reg_verified_id = clean_id
                            sync_patient(clean_id, patient_record)
                            st.success("Patient registered successfully.")
                        else:
                            st.error("Verification failed: Patient record was not found in SQLite database.")
                    else:
                        st.error("Database registration failed. Please check repository connection.")

    # Action after verified registration
    if st.session_state.get('reg_verified_id'):
        reg_id_disp = st.session_state.reg_verified_id
        st.markdown(f"""
        <div style="background: #ecfdf5; border: 1px solid #a7f3d0; border-radius: 12px; padding: 16px; margin: 15px 0;">
            <p style="margin: 0; color: #065f46; font-size: 0.95rem;">
                <strong>Verified Patient Record:</strong> Patient <code>{reg_id_disp}</code> is committed to the local SQLite database (<code>clinic_local.db</code>).
            </p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Patient", key="btn_open_newly_registered"):
            sync_patient(reg_id_disp)
            st.session_state.reg_verified_id = None
            st.session_state.current_nav = "Patient Search and Selection"
            st.rerun()

# -----------------------------------------------------------------------------
# PAGE 3: PATIENT SEARCH AND SELECTION
# -----------------------------------------------------------------------------
elif selected_nav == "Patient Search and Selection":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Patient Search and Selection</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Search existing patient records from the local database and activate their diagnostic workspace.</p>
    </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("🔍 Search by Patient ID or Patient Name", placeholder="Type ID or Name...")

    p_repo_inst = PatientRepository('clinic_local.db')
    v_repo_inst = VisitRepository('clinic_local.db')
    all_patients = p_repo_inst.list_all_patients()

    if search_query:
        sq = search_query.lower()
        filtered = [
            p for p in all_patients
            if sq in p['patient_id'].lower() or sq in p.get('patient_name', '').lower()
        ]
    else:
        filtered = all_patients

    if not filtered:
        st.warning("No matching patient records found.")
    else:
        patient_rows = []
        for p in filtered:
            pid = p['patient_id']
            pname = p.get('patient_name', pid) or pid
            visits = v_repo_inst.get_visit_history(pid)
            
            last_eval = "No evaluations"
            last_pred = "N/A"
            if visits:
                last_v = visits[-1]
                last_eval = last_v.get('visit_timestamp', 'N/A')
                p_class = last_v.get('predicted_class')
                if p_class is not None:
                    last_pred = "PNEUMONIA" if p_class == 1 else "NORMAL"

            obs = visits[-1] if visits else {}
            patient_rows.append({
                'Patient ID': pid,
                'Patient Name': pname,
                'Age': f"{obs.get('age', 'N/A')} yrs" if obs.get('age') is not None else "N/A",
                'Gender': obs.get('gender', 'N/A'),
                'Last Evaluation': last_eval,
                'Last Prediction': last_pred,
                'Status': p.get('status', 'ACTIVE')
            })

        df_display = pd.DataFrame(patient_rows)
        st.dataframe(df_display, width="stretch")

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'/>", unsafe_allow_html=True)
        
        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            active_default = st.session_state.selected_patient_id or st.session_state.active_patient_id
            filtered_ids = [p['patient_id'] for p in filtered]
            default_index = filtered_ids.index(active_default) if active_default in filtered_ids else 0
            
            selected_id = st.selectbox(
                "Select Patient to Open in Workspace",
                filtered_ids,
                index=default_index,
                format_func=lambda x: f"{x} - {next((p.get('patient_name', x) for p in filtered if p['patient_id'] == x), x)}"
            )
        with col_btn:
            st.write("")
            st.write("")
            if st.button("Open Patient", width="stretch", key="btn_open_patient_search"):
                sync_patient(selected_id)
                # Clear previous temporary analysis for new patient context
                st.session_state.selected_xray = None
                st.session_state.current_evaluation = None
                st.session_state.current_cdss_result = None
                st.session_state.xai_outputs = None
                st.session_state.diagnostic_results = None
                st.success(f"Patient '{selected_id}' is now open in the clinical workspace.")
                time.sleep(0.3)
                st.rerun()

# -----------------------------------------------------------------------------
# PAGE 4: X-RAY ANALYSIS
# -----------------------------------------------------------------------------
elif selected_nav == "X-Ray Analysis":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Radiological X-Ray Analysis Workspace</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Upload or select radiological chest X-ray images for automated multimodal DenseNet-151 inference.</p>
    </div>
    """, unsafe_allow_html=True)

    active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
    if not active_pid:
        st.warning("⚠️ No active patient selected. Please select a patient in 'Patient Search and Selection' or register a new patient first.")
    else:
        # Check presets
        preset_options = ["-- Upload New Custom Image --"]
        preset_paths = []
        if os.path.exists("reports/uploaded_xray.jpeg"):
            preset_options.append("Hospital Preset: Sample Pediatric Chest Scan (reports/uploaded_xray.jpeg)")
            preset_paths.append("reports/uploaded_xray.jpeg")

        col_left, col_right = st.columns([1.1, 1.2])
        target_image_path = None
        uploaded_bytes = None

        with col_left:
            st.markdown("#### 1. Configure Radiological Scan")
            preset_choice = st.selectbox("Preset Clinical Scans", preset_options)
            
            st.markdown("<p style='text-align: center; color: #94a3b8; margin: 10px 0;'>— OR UPLOAD IMAGE —</p>", unsafe_allow_html=True)
            uploaded_file = st.file_uploader("Upload Chest X-ray (PNG, JPG, JPEG)", type=['png', 'jpg', 'jpeg'])

            if uploaded_file is not None:
                uploaded_bytes = uploaded_file.read()
            elif preset_choice != "-- Upload New Custom Image --":
                target_image_path = preset_paths[preset_options.index(preset_choice) - 1]

        with col_right:
            st.markdown("#### 2. Image Preview & Validation")
            if uploaded_bytes:
                img_preview = Image.open(uploaded_file)
                st.image(img_preview, caption=f"Uploaded Image: {uploaded_file.name}", width="stretch")
            elif target_image_path and os.path.exists(target_image_path):
                st.image(target_image_path, caption=f"Preset Image: {target_image_path}", width="stretch")
            else:
                st.markdown("""
                <div class="med-card" style="text-align: center; padding: 40px; color: #94a3b8;">
                    <span style="font-size: 2.5rem;">🩻</span><br>
                    No chest X-ray loaded yet.<br>
                    Please upload an image file or choose a preset test scan.
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'/>", unsafe_allow_html=True)

        if st.button("🚀 Analyze X-Ray with HydroFed CDSS", width="stretch"):
            if not uploaded_bytes and not target_image_path:
                st.error("Please configure an X-ray image (upload or preset) before analyzing.")
            else:
                with st.spinner("Executing BMTF-IIFR preprocessing, DenseNet-151, AICA cross-attention & MC Dropout..."):
                    os.makedirs('reports', exist_ok=True)
                    os.makedirs('reports/xai_outputs', exist_ok=True)
                    target_file = 'reports/uploaded_xray.jpeg'
                    
                    if uploaded_bytes:
                        with open(target_file, 'wb') as f:
                            f.write(uploaded_bytes)
                    elif target_image_path and target_image_path != target_file:
                        with open(target_image_path, 'rb') as f_src, open(target_file, 'wb') as f_dst:
                            f_dst.write(f_src.read())

                    st.session_state.selected_xray = target_file

                    # Prepare Active 5-D Clinical Input Vector
                    data_dem = st.session_state.active_patient_data or {}
                    age = float(data_dem.get('age', 5.0))
                    gender = data_dem.get('gender', 'Male')
                    diabetes = int(data_dem.get('diabetes', 0))
                    smoke = int(data_dem.get('passive_smoke_exposure', 0))
                    family = int(data_dem.get('family_respiratory_history', 0))

                    age_norm = age / 80.0
                    gender_val = 1.0 if gender == 'Female' else 0.0
                    clinical_vector = [age_norm, gender_val, float(diabetes), float(smoke), float(family)]

                    pat_id = active_pid
                    visit_id = f"{pat_id}-V{int(time.time()) % 1000:03d}"

                    # Run Real Inference Service
                    inf_service = get_inference_service()
                    res = inf_service.run_cdss_diagnostics(
                        image_path=target_file,
                        clinical_vector=clinical_vector,
                        output_vis_dir='reports/xai_outputs',
                        visit_id=visit_id
                    )

                    # Save preprocessed image preview
                    prep_vis_path = f"reports/xai_outputs/{visit_id}_preprocessed.png"
                    try:
                        raw_img = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
                        if raw_img is not None:
                            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                            clahe_img = clahe.apply(raw_img)
                            h, w = clahe_img.shape[:2]
                            scale = min(224 / w, 224 / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            resized = cv2.resize(clahe_img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                            padded = np.zeros((224, 224), dtype=np.uint8)
                            padded[(224 - new_h)//2 : (224 - new_h)//2 + new_h, (224 - new_w)//2 : (224 - new_w)//2 + new_w] = resized
                            cv2.imwrite(prep_vis_path, padded)
                        else:
                            prep_vis_path = target_file
                    except Exception:
                        prep_vis_path = target_file

                    # Commit to SQLite Database
                    v_repo_inst = VisitRepository('clinic_local.db')
                    a_repo_inst = AuditRepository('clinic_local.db')
                    
                    v_repo_inst.create_visit(visit_id, pat_id, 'Client-01', 'v1.0')
                    v_repo_inst.store_clinical_observation(
                        visit_id, age, gender, diabetes, smoke, family,
                        height=data_dem.get('height', 0.0),
                        weight=data_dem.get('weight', 0.0),
                        blood_pressure=data_dem.get('blood_pressure', ''),
                        blood_sugar=data_dem.get('blood_sugar', 0.0)
                    )
                    v_repo_inst.store_inference(
                        visit_id=visit_id,
                        predicted_class=res['predicted_class'],
                        prob=res['pneumonia_probability'],
                        confidence=res['confidence'],
                        uncertainty=res['uncertainty'],
                        latency=res['total_cdss_latency']
                    )
                    v_repo_inst.store_xai_result(visit_id, 'Grad-CAM', res['gradcam_heatmap_path'])
                    v_repo_inst.store_xai_result(visit_id, 'Grad-CAM++', res['gradcam_plus_heatmap_path'])
                    a_repo_inst.log_event('INFERENCE_COMPLETED', st.session_state.clinician_name, pat_id, visit_id)

                    # Save in all session state keys
                    diag_payload = {
                        'visit_id': visit_id,
                        'predicted_class': res['predicted_class'],
                        'pneumonia_probability': res['pneumonia_probability'],
                        'confidence': res['confidence'],
                        'uncertainty': res['uncertainty'],
                        'total_cdss_latency': res['total_cdss_latency'],
                        'raw_image_path': target_file,
                        'preprocessed_image_path': prep_vis_path,
                        'gradcam_path': res['gradcam_heatmap_path'],
                        'gradcam_plus_path': res['gradcam_plus_heatmap_path'],
                        'clinical_data': data_dem
                    }

                    st.session_state.diagnostic_results = diag_payload
                    st.session_state.current_evaluation = diag_payload
                    st.session_state.current_cdss_result = diag_payload
                    st.session_state.xai_outputs = {
                        'raw_image': target_file,
                        'gradcam': res['gradcam_heatmap_path'],
                        'gradcam_plus': res['gradcam_plus_heatmap_path']
                    }

                    st.success("✅ Diagnostic analysis complete! Results stored in local SQLite database.")

                    c_nav1, c_nav2 = st.columns(2)
                    with c_nav1:
                        if st.button("View CDSS Classifier Results", width="stretch"):
                            st.session_state.current_nav = "CDSS Classifier"
                            st.rerun()
                    with c_nav2:
                        if st.button("View Pathology Salience (XAI)", width="stretch"):
                            st.session_state.current_nav = "Pathology Salience (XAI)"
                            st.rerun()

# -----------------------------------------------------------------------------
# PAGE 5: ACTIVE CLINICAL INPUTS
# -----------------------------------------------------------------------------
elif selected_nav == "Active Clinical Inputs":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Active Clinical Inputs</h2>
        <p style="color: #64748b; font-size: 0.9rem;">
            Differentiates between complete stored EHR records and the active 5-dimensional feature vector consumed by the HydroFed neural model.
        </p>
    </div>
    """, unsafe_allow_html=True)

    active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
    if not active_pid:
        st.warning("⚠️ No active patient profile loaded. Please select or register a patient first.")
    else:
        data = st.session_state.active_patient_data or {}
        
        st.markdown("#### 1. Active 5-Dimensional AI Input Features")
        st.write("These 5 clinical tokens are mapped by the ClinicalEncoder into 128-dimensional embedding space for cross-attention with DenseNet visual representations.")

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">AGE TOKEN</div>
                <div class="card-stat-value" style="color: #2563eb;">{data.get('age', 'N/A')}</div>
                <div class="card-subtext">Norm: {float(data.get('age', 5.0))/80.0:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">GENDER TOKEN</div>
                <div class="card-stat-value" style="color: #7c3aed;">{data.get('gender', 'N/A')}</div>
                <div class="card-subtext">Val: {1.0 if data.get('gender') == 'Female' else 0.0}</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            diab_val = data.get('diabetes', 0)
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">DIABETES</div>
                <div class="card-stat-value" style="color: {'#dc2626' if diab_val else '#059669'};">{'Present' if diab_val else 'Absent'}</div>
                <div class="card-subtext">Token: {float(diab_val)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            smoke_val = data.get('passive_smoke_exposure', 0)
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">SMOKE EXPOSURE</div>
                <div class="card-stat-value" style="color: {'#d97706' if smoke_val else '#059669'};">{'Yes' if smoke_val else 'No'}</div>
                <div class="card-subtext">Token: {float(smoke_val)}</div>
            </div>
            """, unsafe_allow_html=True)
        with col5:
            fam_val = data.get('family_respiratory_history', 0)
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">FAMILY RESP. HIST.</div>
                <div class="card-stat-value" style="color: {'#dc2626' if fam_val else '#059669'};">{'Positive' if fam_val else 'Negative'}</div>
                <div class="card-subtext">Token: {float(fam_val)}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #e2e8f0; margin: 20px 0;'/>", unsafe_allow_html=True)
        st.markdown("#### 2. Stored Clinical Baseline Record (EHR)")
        
        ehr_df = pd.DataFrame([{
            'Patient ID': active_pid,
            'Age': f"{data.get('age', 'N/A')} years",
            'Gender': data.get('gender', 'N/A'),
            'Height': f"{data.get('height', 'N/A')} cm",
            'Weight': f"{data.get('weight', 'N/A')} kg",
            'Blood Pressure': data.get('blood_pressure', 'N/A'),
            'Blood Sugar': f"{data.get('blood_sugar', 'N/A')} mg/dL",
            'Storage Location': "Local SQLite (clinical_observations)"
        }])
        st.dataframe(ehr_df, width="stretch")

        st.info("💡 **AICA Integration Note**: To ensure weight compatibility across all 60 decentralized clinic nodes without dimensionality drift, the active AI clinical vector strictly uses the 5-dimensional formulation [Age, Gender, Diabetes, Smoke, Family History]. Additional EHR fields are retained locally for clinician review.")

# -----------------------------------------------------------------------------
# PAGE 6: CDSS CLASSIFIER
# -----------------------------------------------------------------------------
elif selected_nav == "CDSS Classifier":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">CDSS Classifier</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Diagnostic prediction scores, MC Dropout predictive uncertainty, and clinician review signing.</p>
    </div>
    """, unsafe_allow_html=True)

    res = st.session_state.current_cdss_result or st.session_state.diagnostic_results
    if not res:
        st.warning("⚠️ No active CDSS diagnostic results found. Please run an evaluation on the 'X-Ray Analysis' page first.")
    else:
        visit_id = res['visit_id']

        is_pneu = res['predicted_class'] == 1
        badge_class = "badge-red" if is_pneu else "badge-green"
        pred_label = "PNEUMONIA" if is_pneu else "NORMAL"

        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 16px; border: 1px solid #e2e8f0; padding: 18px 24px; margin-bottom: 22px; box-shadow: 0 4px 12px rgba(15, 23, 42, 0.03);">
            <div style="font-size: 0.8rem; font-weight: 800; letter-spacing: 1px; color: #2563eb; text-transform: uppercase;">AI-ASSISTED CDSS RESULT</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin-top: 4px;">
                Prediction: <span class="badge-status {badge_class}" style="font-size: 1.25rem; vertical-align: middle;">{pred_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">PREDICTION</div>
                <div style="margin: 10px 0;"><span class="badge-status {badge_class}" style="font-size: 1.15rem; padding: 6px 16px;">{pred_label}</span></div>
                <div class="card-subtext">AI-assisted diagnosis</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">PNEUMONIA PROBABILITY</div>
                <div class="card-stat-value" style="color: #dc2626;">{res['pneumonia_probability']*100:.2f}%</div>
                <div class="card-subtext">Calibrated Softmax index</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">CONFIDENCE</div>
                <div class="card-stat-value" style="color: #059669;">{res['confidence']*100:.2f}%</div>
                <div class="card-subtext">Mean ensemble score</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="med-card" style="text-align: center;">
                <div class="card-header-label">UNCERTAINTY (MC DROPOUT)</div>
                <div class="card-stat-value" style="color: #d97706;">{res['uncertainty']:.4f}</div>
                <div class="card-subtext">Inference Latency: {res['total_cdss_latency']*1000:.1f} ms</div>
            </div>
            """, unsafe_allow_html=True)

        # Safety Uncertainty Boundary Check
        if res['uncertainty'] > 0.15:
            st.markdown("""
            <div style="background: #fef2f2; border: 2px solid #ef4444; border-radius: 12px; padding: 16px; margin-bottom: 20px;">
                <h4 style="color: #dc2626; margin: 0 0 4px 0;">🚨 CLINICIAN REVIEW REQUIRED</h4>
                <p style="color: #991b1b; margin: 0; font-size: 0.92rem;">
                    Predictive uncertainty exceeds nominal safety boundary (0.1500). The model indicates high variance across stochastic passes. Do not rely on automated indices without radiologist corroboration.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #ecfdf5; border: 1px solid #10b981; border-radius: 12px; padding: 14px; margin-bottom: 20px;">
                <span style="color: #065f46; font-weight: 700;">✅ Uncertainty Index Within Nominal Limits (&le; 0.1500).</span>
            </div>
            """, unsafe_allow_html=True)

        # Mandatory Clinical Disclaimer
        st.info("⚠️ **Clinical Disclaimer**: This result is intended to support clinical assessment and does not replace professional clinical judgment. This system does not prescribe autonomous clinical treatments or medications.")

        # Visual Comparison: Raw vs Preprocessed
        c_v1, c_v2 = st.columns(2)
        with c_v1:
            st.markdown("#### Input Radiological Scan")
            if os.path.exists(res.get('raw_image_path', '')):
                st.image(res['raw_image_path'], width="stretch")
        with c_v2:
            st.markdown("#### Contrast Enhanced (CLAHE Preprocessed)")
            if os.path.exists(res.get('preprocessed_image_path', '')):
                st.image(res['preprocessed_image_path'], width="stretch")

        st.markdown("<hr style='border-color: #e2e8f0; margin: 24px 0;'/>", unsafe_allow_html=True)

        # Clinician Review & Signing Form
        st.markdown("### Clinician Diagnostic Review & SQLite Signing")
        with st.form("clinician_review_form"):
            rev_ref = st.text_input("Clinician Reference ID / Name Signature *", value=st.session_state.clinician_name)
            rev_status = st.radio("Diagnostic Review Decision *", ["CONFIRMED", "NEEDS_REVIEW", "UNABLE_TO_DETERMINE"])
            rev_notes = st.text_area("Clinical Notes & Diagnostic Remarks", placeholder="Enter structural findings, lobar infiltrates, consolidation remarks...")
            
            signed = st.form_submit_button("Sign & Save Diagnostic Record to Database")
            if signed:
                if not rev_ref.strip():
                    st.error("Clinician signature reference is mandatory.")
                else:
                    v_repo_inst = VisitRepository('clinic_local.db')
                    a_repo_inst = AuditRepository('clinic_local.db')
                    success = v_repo_inst.store_clinician_review(visit_id, rev_status, rev_ref.strip(), rev_notes.strip())
                    if success:
                        active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
                        a_repo_inst.log_event('CLINICIAN_REVIEWED', rev_ref.strip(), active_pid, visit_id)
                        st.success(f"Review for visit '{visit_id}' signed and logged in local SQLite audit trail!")
                    else:
                        st.error("Failed to commit clinician review.")

# -----------------------------------------------------------------------------
# PAGE 7: PATHOLOGY SALIENCE (XAI)
# -----------------------------------------------------------------------------
elif selected_nav == "Pathology Salience (XAI)":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Pathology Salience (XAI)</h2>
        <p style="color: #64748b; font-size: 0.9rem;">
            Explainability heatmaps generated via Grad-CAM and Grad-CAM++ targeting the final DenseNet-151 normalization layer.
        </p>
    </div>
    """, unsafe_allow_html=True)

    res = st.session_state.current_cdss_result or st.session_state.diagnostic_results
    if not res:
        st.warning("⚠️ No active explainability outputs available. Please execute an X-ray diagnostic analysis first.")
    else:
        raw_p = res.get('raw_image_path', '')
        gcam_p = res.get('gradcam_path', '')
        gcam_pp = res.get('gradcam_plus_path', '')

        st.info("ℹ️ **Explainability Notice**: Highlighted regions represent image areas that contributed to the model prediction. Explainability visualizations illustrate network gradient salience and do not represent confirmed anatomical lesion locations.")

        # Responsive columns for Original X-ray, Grad-CAM, and Grad-CAM++
        col_orig, col_x1, col_x2 = st.columns(3)
        with col_orig:
            st.markdown("#### Original X-ray")
            if os.path.exists(raw_p):
                st.image(raw_p, caption="Original Patient Radiograph", width="stretch")
            else:
                st.write("Original X-ray image file not found.")

        with col_x1:
            st.markdown("#### Grad-CAM")
            if os.path.exists(gcam_p):
                st.image(gcam_p, caption="Grad-CAM (DenseNet-151 Final Norm)", width="stretch")
            else:
                st.write("Grad-CAM visualization is not available for this evaluation.")

        with col_x2:
            st.markdown("#### Grad-CAM++")
            if os.path.exists(gcam_pp):
                st.image(gcam_pp, caption="Grad-CAM++ (Higher-Order Gradient Sensitivity)", width="stretch")
            else:
                st.write("Grad-CAM++ visualization is not available for this evaluation.")

# -----------------------------------------------------------------------------
# PAGE 8: LONGITUDINAL TREND
# -----------------------------------------------------------------------------
elif selected_nav == "Longitudinal Trend":
    render_top_header()
    render_active_patient_banner()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Longitudinal Trend</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Temporal disease progression tracking and clinical stability analysis across historical clinic visits.</p>
    </div>
    """, unsafe_allow_html=True)

    active_pid = st.session_state.selected_patient_id or st.session_state.active_patient_id
    if not active_pid:
        st.warning("⚠️ Please select an active patient to inspect longitudinal records.")
    else:
        v_repo_inst = VisitRepository('clinic_local.db')
        visits = v_repo_inst.get_visit_history(active_pid)

        if not visits:
            st.markdown("""
            <div class="med-card" style="text-align: center; padding: 40px; color: #64748b;">
                <span style="font-size: 2rem;">📈</span><br>
                <strong>No previous evaluations available.</strong><br>
                Perform evaluations on the X-Ray Analysis page to build a temporal clinical timeline.
            </div>
            """, unsafe_allow_html=True)
        else:
            c_chart, c_engine = st.columns([1.3, 1.0])
            with c_chart:
                st.markdown("#### Probability & Uncertainty Over Time")
                setup_light_matplotlib()
                fig, ax = plt.subplots(figsize=(6.5, 4))

                dates = [pd.to_datetime(v['visit_timestamp']) for v in visits]
                probs = [v['pneumonia_probability'] if v['pneumonia_probability'] is not None else 0.0 for v in visits]
                confs = [v['confidence'] if v['confidence'] is not None else 0.0 for v in visits]
                uncs = [v['uncertainty'] if v['uncertainty'] is not None else 0.0 for v in visits]

                ax.plot(dates, probs, marker='o', color='#ef4444', linewidth=2.2, label='Pneumonia Prob.')
                ax.plot(dates, confs, marker='s', color='#10b981', linewidth=1.8, linestyle='--', label='Confidence')
                ax.plot(dates, uncs, marker='^', color='#f59e0b', linewidth=1.8, linestyle=':', label='Uncertainty')

                ax.set_ylim(0.0, 1.05)
                ax.set_ylabel("Index Score")
                ax.grid(True, alpha=0.3, color='#e2e8f0')
                ax.legend(frameon=True, facecolor='#ffffff', edgecolor='#e2e8f0')
                fig.autofmt_xdate()
                st.pyplot(fig)

            with c_engine:
                st.markdown("#### Longitudinal Engine Trend Assessment")
                engine = LongitudinalTrendEngine()
                trend_res = engine.analyze_trend(visits)
                
                trend_val = trend_res['trend']
                trend_badge = "badge-blue"
                if trend_val == 'IMPROVING':
                    trend_badge = "badge-green"
                elif trend_val == 'WORSENING':
                    trend_badge = "badge-red"
                elif trend_val == 'UNCERTAIN':
                    trend_badge = "badge-amber"

                st.markdown(f"""
                <div class="med-card">
                    <div class="card-header-label">TEMPORAL ASSESSMENT</div>
                    <div style="margin: 10px 0;"><span class="badge-status {trend_badge}" style="font-size: 1.1rem; padding: 6px 16px;">{trend_val}</span></div>
                    <p style="color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0;">{trend_res['message']}</p>
                </div>
                """, unsafe_allow_html=True)

                if trend_res.get('warning'):
                    st.warning(f"⚠️ {trend_res['warning']}")

            st.markdown("#### Historical Evaluation Timeline")
            df_hist = pd.DataFrame(visits)
            if not df_hist.empty:
                cols_to_show = ['visit_timestamp', 'predicted_class', 'pneumonia_probability', 'confidence', 'uncertainty', 'review_status', 'clinical_note']
                cols_exist = [c for c in cols_to_show if c in df_hist.columns]
                df_sub = df_hist[cols_exist].copy()
                df_sub.rename(columns={
                    'visit_timestamp': 'Visit Date',
                    'predicted_class': 'Prediction (0=Norm, 1=Pneu)',
                    'pneumonia_probability': 'Probability',
                    'confidence': 'Confidence',
                    'uncertainty': 'Uncertainty',
                    'review_status': 'Clinician Review',
                    'clinical_note': 'Notes'
                }, inplace=True)
                st.dataframe(df_sub, width="stretch")

# -----------------------------------------------------------------------------
# PAGE 9: EDGE PERFORMANCE METRICS
# -----------------------------------------------------------------------------
elif selected_nav == "Edge Performance Metrics":
    render_top_header()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">Edge Performance Metrics</h2>
        <p style="color: #64748b; font-size: 0.9rem;">Real-time host system resource telemetry and verified edge quantization benchmarks.</p>
    </div>
    """, unsafe_allow_html=True)

    # Live System Telemetry
    cpu_percent = psutil.cpu_percent(interval=None)
    mem_info = psutil.virtual_memory()
    proc = psutil.Process(os.getpid())
    proc_mem = proc.memory_info().rss / (1024 * 1024)

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">CPU UTILIZATION</div>
            <div class="card-stat-value" style="color: #2563eb;">{cpu_percent:.1f}%</div>
            <div class="card-subtext">Active Host Edge Cores: {psutil.cpu_count(logical=True)}</div>
        </div>
        """, unsafe_allow_html=True)
    with t2:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">AVAILABLE RAM</div>
            <div class="card-stat-value" style="color: #059669;">{mem_info.available / (1024**3):.1f} GB</div>
            <div class="card-subtext">Total System: {mem_info.total / (1024**3):.1f} GB</div>
        </div>
        """, unsafe_allow_html=True)
    with t3:
        st.markdown(f"""
        <div class="med-card">
            <div class="card-header-label">PROCESS RSS MEMORY</div>
            <div class="card-stat-value" style="color: #7c3aed;">{proc_mem:.1f} MB</div>
            <div class="card-subtext">CDSS Python Process</div>
        </div>
        """, unsafe_allow_html=True)
    with t4:
        st.markdown("""
        <div class="med-card">
            <div class="card-header-label">EXECUTION TARGET</div>
            <div class="card-stat-value" style="color: #0f172a; font-size: 1.45rem;">Edge CPU</div>
            <div class="card-subtext">Quantized Int8 Ready</div>
        </div>
        """, unsafe_allow_html=True)

    # Recorded Benchmarks Table
    st.markdown("#### Quantization & Model Distillation Benchmarks")
    edge_table_csv = "IEEE_RESULTS/22_FINAL_PAPER_DATA/table_07_edge.csv"
    if os.path.exists(edge_table_csv):
        df_bench = pd.read_csv(edge_table_csv)
        st.dataframe(df_bench, width="stretch")
    else:
        st.write("Benchmark data not available.")

    # Hardware Profile
    st.markdown("#### Edge Host Specifications")
    hw_file = "IEEE_RESULTS/hardware.txt"
    if os.path.exists(hw_file):
        with open(hw_file, 'r') as f:
            hw_content = f.read()
        st.code(hw_content, language="yaml")

# -----------------------------------------------------------------------------
# PAGE 10: SYSTEM SECURITY AUDIT LOG
# -----------------------------------------------------------------------------
elif selected_nav == "System Security Audit Log":
    render_top_header()

    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h2 style="font-size: 1.6rem; font-weight: 800; color: #0f172a; margin: 0;">System Security Audit Log</h2>
        <p style="color: #64748b; font-size: 0.9rem;">
            Immutable audit records for clinician logins, patient registrations, CDSS diagnostic passes, and review signatures.
        </p>
    </div>
    """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="med-card">
            <div class="card-header-label">ENCRYPTION PROTOCOL</div>
            <div class="card-stat-value" style="font-size: 1.35rem; color: #2563eb;">AES-256-GCM</div>
            <div class="card-subtext">Authenticated confidential updates</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="med-card">
            <div class="card-header-label">PATIENT DATA PRIVACY</div>
            <div class="card-stat-value" style="font-size: 1.35rem; color: #059669;">LOCAL ONLY</div>
            <div class="card-subtext">Zero cloud or remote data leakage</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="med-card">
            <div class="card-header-label">SIGNATURE VALIDATION</div>
            <div class="card-stat-value" style="font-size: 1.35rem; color: #7c3aed;">HMAC / SHA-256</div>
            <div class="card-subtext">Timing-safe cryptographic checks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("#### Immutable Database Audit Events")
    a_repo_inst = AuditRepository('clinic_local.db')
    logs = a_repo_inst.get_logs(limit=100)

    if not logs:
        st.write("No audit events recorded.")
    else:
        df_audit = pd.DataFrame(logs)
        df_audit.rename(columns={
            'timestamp': 'Timestamp',
            'event_type': 'Event Type',
            'actor': 'Clinician / Actor',
            'patient_id': 'Patient Reference',
            'visit_id': 'Visit Reference'
        }, inplace=True)
        cols_disp = ['Timestamp', 'Event Type', 'Clinician / Actor', 'Patient Reference', 'Visit Reference']
        st.dataframe(df_audit[[c for c in cols_disp if c in df_audit.columns]], width="stretch")
