"""
HydroFed-ICAF Clinical Report Generator
Generates a complete PDF clinical decision-support report using ReportLab.
All data must come from real backend outputs — no fake/hardcoded values.
"""
import os
import io
import time
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.colors import HexColor
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
    HRFlowable, PageBreak, KeepTogether
)
from reportlab.lib import colors


# ─── Color palette ────────────────────────────────────────────────────────────
PRIMARY_BLUE = HexColor('#1d4ed8')
DARK_TEXT = HexColor('#0f172a')
GREY_TEXT = HexColor('#475569')
LIGHT_GREY = HexColor('#f1f5f9')
BORDER_GREY = HexColor('#cbd5e1')
GREEN = HexColor('#059669')
RED = HexColor('#dc2626')
AMBER = HexColor('#d97706')
WHITE = HexColor('#ffffff')


def _build_styles():
    """Creates all paragraph styles used in the report."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'ReportTitle', parent=styles['Title'],
        fontSize=20, textColor=PRIMARY_BLUE, spaceAfter=4, alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    ))
    styles.add(ParagraphStyle(
        'ReportSubtitle', parent=styles['Normal'],
        fontSize=11, textColor=GREY_TEXT, alignment=TA_CENTER,
        spaceAfter=14, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'SectionHeading', parent=styles['Heading2'],
        fontSize=13, textColor=PRIMARY_BLUE, spaceBefore=16, spaceAfter=8,
        fontName='Helvetica-Bold', borderWidth=0, leftIndent=0
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=9.5, textColor=DARK_TEXT, leading=14, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'SmallGrey', parent=styles['Normal'],
        fontSize=8, textColor=GREY_TEXT, leading=11, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'Disclaimer', parent=styles['Normal'],
        fontSize=8, textColor=GREY_TEXT, leading=11, fontName='Helvetica-Oblique',
        spaceBefore=10, alignment=TA_LEFT
    ))
    styles.add(ParagraphStyle(
        'TableCell', parent=styles['Normal'],
        fontSize=9, textColor=DARK_TEXT, leading=12, fontName='Helvetica'
    ))
    styles.add(ParagraphStyle(
        'TableHeader', parent=styles['Normal'],
        fontSize=9, textColor=WHITE, leading=12, fontName='Helvetica-Bold'
    ))
    return styles


def _make_section_header(text, styles):
    return Paragraph(text, styles['SectionHeading'])


def _make_kv_table(data_pairs, styles, col_widths=None):
    """Creates a two-column key-value table."""
    if not col_widths:
        col_widths = [55 * mm, 95 * mm]

    table_data = []
    for key, value in data_pairs:
        table_data.append([
            Paragraph(f"<b>{key}</b>", styles['TableCell']),
            Paragraph(str(value) if value is not None else "N/A", styles['TableCell'])
        ])

    t = Table(table_data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), LIGHT_GREY),
        ('TEXTCOLOR', (0, 0), (-1, -1), DARK_TEXT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t


def _try_add_image(path, max_width=65 * mm, max_height=65 * mm):
    """Attempts to add an image from disk. Returns RLImage or None."""
    if not path or not os.path.exists(path):
        return None
    try:
        img = RLImage(path)
        iw, ih = img.drawWidth, img.drawHeight
        if iw <= 0 or ih <= 0:
            return None
        scale = min(max_width / iw, max_height / ih, 1.0)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return img
    except Exception:
        return None


def generate_clinical_report(
    patient_info,
    patient_data,
    diagnostic_result,
    visit_history=None,
    clinician_name="",
    xray_path=None,
    gradcam_path=None,
    gradcam_plus_path=None
):
    """
    Generates a complete HydroFed-ICAF clinical PDF report in memory.

    Args:
        patient_info: dict with 'patient_id', 'patient_name', etc. from PatientRepository
        patient_data: dict with clinical fields ('age', 'gender', 'diabetes', etc.)
        diagnostic_result: dict from InferenceService / session state with prediction data
        visit_history: list of dicts from VisitRepository.get_visit_history()
        clinician_name: str — the logged-in clinician
        xray_path: str — path to the analyzed X-ray image
        gradcam_path: str — path to Grad-CAM visualization
        gradcam_plus_path: str — path to Grad-CAM++ visualization

    Returns:
        (bytes, filename) tuple — PDF bytes and suggested filename, or (None, error_message)
    """
    # ── Validate required data ────────────────────────────────────────────
    if not patient_info:
        return None, "Unable to generate report: patient information is unavailable."
    if not diagnostic_result:
        return None, "Unable to generate report: no analysis result is available."

    pid = patient_info.get('patient_id', 'UNKNOWN')
    pname = patient_info.get('patient_name', pid)
    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    file_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"HydroFed_ICAF_Report_{pid}_{file_ts}.pdf"

    styles = _build_styles()
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm
    )
    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("HYDROFED-ICAF", styles['ReportTitle']))
    story.append(Paragraph("Clinical Decision Support System — Analysis Report", styles['ReportSubtitle']))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY_BLUE, spaceAfter=6))
    story.append(Paragraph(f"Generated: {timestamp_str}&nbsp;&nbsp;|&nbsp;&nbsp;Clinician: {clinician_name or 'N/A'}", styles['SmallGrey']))
    story.append(Spacer(1, 8))

    # ── Patient Information ───────────────────────────────────────────────
    story.append(_make_section_header("Patient Information", styles))
    patient_pairs = [
        ("Patient ID", pid),
        ("Patient Name", pname),
        ("Age", f"{patient_data.get('age', 'N/A')} years" if patient_data else "N/A"),
        ("Gender", patient_data.get('gender', 'N/A') if patient_data else "N/A"),
        ("Status", patient_info.get('status', 'ACTIVE')),
    ]
    story.append(_make_kv_table(patient_pairs, styles))
    story.append(Spacer(1, 6))

    # ── Clinical Inputs ───────────────────────────────────────────────────
    if patient_data:
        story.append(_make_section_header("Clinical Inputs (5-D Model Vector)", styles))

        def _bool_label(val, yes='Present', no='Absent'):
            return yes if val else no

        clinical_pairs = [
            ("Age (Normalized)", f"{float(patient_data.get('age', 0)) / 80.0:.3f}"),
            ("Gender", patient_data.get('gender', 'N/A')),
            ("Diabetes", _bool_label(patient_data.get('diabetes', 0))),
            ("Passive Smoke Exposure", _bool_label(patient_data.get('passive_smoke_exposure', 0), 'Yes', 'No')),
            ("Family Respiratory History", _bool_label(patient_data.get('family_respiratory_history', 0), 'Positive', 'Negative')),
        ]

        # Additional vitals if available
        if patient_data.get('height'):
            clinical_pairs.append(("Height", f"{patient_data['height']} cm"))
        if patient_data.get('weight'):
            clinical_pairs.append(("Weight", f"{patient_data['weight']} kg"))
        if patient_data.get('blood_pressure'):
            clinical_pairs.append(("Blood Pressure", f"{patient_data['blood_pressure']} mmHg"))
        if patient_data.get('blood_sugar'):
            clinical_pairs.append(("Fasting Blood Sugar", f"{patient_data['blood_sugar']} mg/dL"))

        story.append(_make_kv_table(clinical_pairs, styles))
        story.append(Spacer(1, 6))

    # ── X-Ray Image ───────────────────────────────────────────────────────
    xray_img = _try_add_image(xray_path, max_width=80 * mm, max_height=80 * mm)
    if xray_img:
        story.append(_make_section_header("Analyzed Chest X-Ray", styles))
        story.append(xray_img)
        story.append(Spacer(1, 6))

    # ── CDSS Result ───────────────────────────────────────────────────────
    story.append(_make_section_header("CDSS Diagnostic Result", styles))
    pred_class = diagnostic_result.get('predicted_class', 'N/A')
    pred_label = "PNEUMONIA" if pred_class == 1 else "NORMAL" if pred_class == 0 else str(pred_class)
    prob = diagnostic_result.get('pneumonia_probability')
    conf = diagnostic_result.get('confidence')
    unc = diagnostic_result.get('uncertainty')
    latency = diagnostic_result.get('total_cdss_latency')

    result_pairs = [
        ("Prediction", pred_label),
        ("Pneumonia Probability", f"{prob * 100:.2f}%" if prob is not None else "N/A"),
        ("Confidence", f"{conf * 100:.2f}%" if conf is not None else "N/A"),
        ("Predictive Uncertainty (MC Dropout)", f"{unc:.4f}" if unc is not None else "N/A"),
        ("Inference Latency", f"{latency * 1000:.1f} ms" if latency is not None else "N/A"),
    ]
    story.append(_make_kv_table(result_pairs, styles))
    story.append(Spacer(1, 6))

    # ── Pneumonia Risk ────────────────────────────────────────────────────
    story.append(_make_section_header("Pneumonia Risk Assessment", styles))
    if prob is not None:
        risk_text = f"Model-estimated pneumonia probability: <b>{prob * 100:.2f}%</b>"
        story.append(Paragraph(risk_text, styles['BodyText2']))
    else:
        story.append(Paragraph("Pneumonia risk data is not available for this analysis.", styles['BodyText2']))

    story.append(Paragraph(
        "This probability represents the model's estimated likelihood and should not be interpreted "
        "as a confirmed clinical diagnosis. Clinical correlation is required.",
        styles['SmallGrey']
    ))
    story.append(Spacer(1, 6))

    # ── XAI / Pathology Salience ──────────────────────────────────────────
    gcam_img = _try_add_image(gradcam_path, max_width=70 * mm, max_height=70 * mm)
    gcam_plus_img = _try_add_image(gradcam_plus_path, max_width=70 * mm, max_height=70 * mm)

    if gcam_img or gcam_plus_img:
        story.append(_make_section_header("XAI / Pathology Salience", styles))

        xai_row = []
        xai_headers = []
        if gcam_img:
            xai_headers.append(Paragraph("<b>Grad-CAM</b>", styles['TableCell']))
            xai_row.append(gcam_img)
        if gcam_plus_img:
            xai_headers.append(Paragraph("<b>Grad-CAM++</b>", styles['TableCell']))
            xai_row.append(gcam_plus_img)

        if xai_row:
            col_w = 80 * mm
            t = Table([xai_headers, xai_row], colWidths=[col_w] * len(xai_row))
            t.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(t)

        story.append(Paragraph(
            "Highlighted regions represent image areas that contributed to the model prediction "
            "and should be interpreted with clinical judgment. These do not represent confirmed "
            "anatomical lesion locations.",
            styles['SmallGrey']
        ))
        story.append(Spacer(1, 6))

    # ── Longitudinal Information ──────────────────────────────────────────
    story.append(_make_section_header("Longitudinal History", styles))
    if visit_history and len(visit_history) > 1:
        # Build history table
        header_row = [
            Paragraph("<b>Date</b>", styles['TableHeader']),
            Paragraph("<b>Prediction</b>", styles['TableHeader']),
            Paragraph("<b>Probability</b>", styles['TableHeader']),
            Paragraph("<b>Confidence</b>", styles['TableHeader']),
            Paragraph("<b>Uncertainty</b>", styles['TableHeader']),
        ]
        hist_data = [header_row]
        for v in visit_history[-10:]:  # Last 10 evaluations
            p_class = v.get('predicted_class')
            p_prob = v.get('pneumonia_probability')
            p_conf = v.get('confidence')
            p_unc = v.get('uncertainty')
            if p_prob is None:
                continue
            p_label = "PNEUMONIA" if p_class == 1 else "NORMAL" if p_class == 0 else "N/A"
            hist_data.append([
                Paragraph(str(v.get('visit_timestamp', 'N/A'))[:19], styles['TableCell']),
                Paragraph(p_label, styles['TableCell']),
                Paragraph(f"{p_prob * 100:.1f}%" if p_prob is not None else "N/A", styles['TableCell']),
                Paragraph(f"{p_conf * 100:.1f}%" if p_conf is not None else "N/A", styles['TableCell']),
                Paragraph(f"{p_unc:.4f}" if p_unc is not None else "N/A", styles['TableCell']),
            ])

        if len(hist_data) > 1:
            col_ws = [38 * mm, 28 * mm, 28 * mm, 28 * mm, 28 * mm]
            ht = Table(hist_data, colWidths=col_ws)
            ht.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), PRIMARY_BLUE),
                ('TEXTCOLOR', (0, 0), (-1, 0), WHITE),
                ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GREY),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [WHITE, LIGHT_GREY]),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 0), (-1, -1), 4),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
                ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ]))
            story.append(ht)
        else:
            story.append(Paragraph("No previous evaluations with prediction data available.", styles['BodyText2']))
    else:
        story.append(Paragraph("No previous evaluations available.", styles['BodyText2']))
    story.append(Spacer(1, 6))

    # ── Edge Performance ──────────────────────────────────────────────────
    story.append(_make_section_header("Edge Performance Metrics", styles))
    edge_pairs = [
        ("Total CDSS Latency", f"{latency * 1000:.1f} ms" if latency is not None else "N/A"),
        ("Execution Target", "Edge CPU"),
    ]
    core_lat = diagnostic_result.get('core_inference_latency')
    if core_lat is not None:
        edge_pairs.insert(1, ("Core Inference Latency", f"{core_lat * 1000:.1f} ms"))
    story.append(_make_kv_table(edge_pairs, styles))
    story.append(Spacer(1, 6))

    # ── Security / Audit ──────────────────────────────────────────────────
    story.append(_make_section_header("Audit Information", styles))
    audit_pairs = [
        ("Analysis Timestamp", timestamp_str),
        ("Clinician", clinician_name or "N/A"),
        ("Visit ID", diagnostic_result.get('visit_id', 'N/A')),
        ("Encryption Protocol", "AES-256-GCM"),
        ("Data Locality", "Local SQLite — No Cloud Transmission"),
    ]
    story.append(_make_kv_table(audit_pairs, styles))
    story.append(Spacer(1, 10))

    # ── Disclaimer ────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER_GREY, spaceAfter=6))
    story.append(Paragraph(
        "This report contains AI-assisted clinical decision-support information generated by the "
        "HydroFed-ICAF system. It is intended to support, not replace, professional clinical assessment. "
        "This system does not prescribe autonomous clinical treatments or medications. All diagnostic "
        "probabilities are model estimates and must be interpreted within the full clinical context by "
        "qualified healthcare professionals.",
        styles['Disclaimer']
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        f"© {datetime.now().year} HydroFed-ICAF Research Prototype",
        styles['SmallGrey']
    ))

    # ── Build PDF ─────────────────────────────────────────────────────────
    try:
        doc.build(story)
        pdf_bytes = buffer.getvalue()
        buffer.close()
        return pdf_bytes, filename
    except Exception as e:
        buffer.close()
        return None, f"PDF generation failed: {str(e)}"
