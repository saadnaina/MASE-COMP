"""
MODULE TEN: Professional Clinical PDF Report Generator using ReportLab.
Produces hospital-style multi-section assessment reports containing:
1. NeuroScan AI Branding & GMU MASE 2026 Header
2. Pseudonymous Patient Reference ID & Timestamp
3. Integrated Urgency Banner & Recommended Next Action
4. FAST & Speech Screening Findings
5. Vital Signs with Provenance Labels
6. Brain CT Analysis Findings & Embedded Grad-CAM / Overlay Images
7. Multimodal Conflicts, Missing Information & Model Uncertainty
8. Mandatory Clinical Disclaimer & Non-Diagnostic Notice
"""
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config import APP_NAME, APP_VERSION, COMPETITION_NAME, MEDICAL_DISCLAIMER_FULL


def _numpy_rgb_to_rl_image(rgb_arr: np.ndarray, width_mm: float = 52.0) -> RLImage:
    pil_img = PILImage.fromarray(rgb_arr.astype(np.uint8))
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    aspect = rgb_arr.shape[0] / max(rgb_arr.shape[1], 1)
    w = width_mm * mm
    h = w * aspect
    return RLImage(buf, width=w, height=h)


def generate_clinical_pdf_report(
    output_pdf_path: Path,
    assessment_bundle: Dict[str, Any],
    integrated_evaluation: Dict[str, Any],
    ct_visuals: Optional[Dict[str, np.ndarray]] = None,
) -> Path:
    """Builds and writes a formatted PDF report to output_pdf_path."""
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=A4,
        leftMargin=16 * mm,
        rightMargin=16 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        title=f"{APP_NAME} Clinical Screening Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "NeuroTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        textColor=colors.HexColor("#0A1128"),
        spaceAfter=2,
    )
    sub_style = ParagraphStyle(
        "NeuroSub",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        textColor=colors.HexColor("#0077B6"),
        spaceAfter=8,
    )
    sec_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=colors.HexColor("#101935"),
        spaceBefore=10,
        spaceAfter=4,
    )
    body_style = ParagraphStyle(
        "BodySmall",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B"),
    )
    disclaimer_style = ParagraphStyle(
        "DisclaimerBox",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#7F1D1D"),
    )

    story = []

    # 1. Header
    story.append(Paragraph(f"{APP_NAME} — Multimodal Stroke Screening Report", title_style))
    story.append(
        Paragraph(
            f"{COMPETITION_NAME} | Model Version: {APP_VERSION} | Generated: "
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
            sub_style,
        )
    )

    # 2. Mandatory Safety Banner
    safety_table = Table(
        [[Paragraph(MEDICAL_DISCLAIMER_FULL, disclaimer_style)]],
        colWidths=[178 * mm],
    )
    safety_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FEE2E2")),
                ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor("#DC2626")),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(safety_table)
    story.append(Spacer(1, 4 * mm))

    # 3. Patient & Assessment Metadata
    asm = assessment_bundle.get("assessment", {})
    pat = assessment_bundle.get("patient", {})
    vit = assessment_bundle.get("vitals", {})
    fast = assessment_bundle.get("fast", {})
    ct = assessment_bundle.get("ct", {})

    is_demo = asm.get("is_demo_record", 0) == 1
    provenance_mode = "SIMULATED_DEMO RECORD" if is_demo else "LIVE SCREENING RECORD"

    meta_data = [
        ["Assessment ID:", asm.get("assessment_id", "N/A"), "Patient Ref ID:", pat.get("patient_ref_id", "N/A")],
        ["Patient Age:", str(pat.get("age", "Not Recorded")), "Symptom Onset:", str(pat.get("symptom_onset_time", "Unknown"))],
        ["Record Provenance:", provenance_mode, "Assessment Timestamp:", str(asm.get("created_at", "N/A"))[:19]],
    ]
    meta_table = Table(meta_data, colWidths=[35 * mm, 54 * mm, 38 * mm, 51 * mm])
    meta_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(meta_table)

    # 4. Integrated Urgency Decision
    story.append(Paragraph("1. Integrated Urgency Assessment & Clinical Recommendation", sec_style))
    urg_level = integrated_evaluation.get("urgency_level", "PENDING")
    rec_action = integrated_evaluation.get("recommended_action", "")
    urg_bg = (
        colors.HexColor("#FECACA")
        if "EMERGENCY" in urg_level
        else colors.HexColor("#FEF3C7")
        if "HIGH_PRIORITY" in urg_level
        else colors.HexColor("#E0F2FE")
    )
    urg_table = Table(
        [
            ["URGENCY CLASSIFICATION:", urg_level],
            ["RECOMMENDED NEXT ACTION:", Paragraph(rec_action, body_style)],
        ],
        colWidths=[48 * mm, 130 * mm],
    )
    urg_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), urg_bg),
                ("BOX", (0, 0), (-1, -1), 1.0, colors.HexColor("#0A1128")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(urg_table)

    # 5. FAST & Vital Signs Table
    story.append(Paragraph("2. FAST Neurological Screening & Vital Signs", sec_style))
    bp_str = (
        f"{vit.get('systolic_bp')}/{vit.get('diastolic_bp')} mmHg"
        if vit.get("systolic_bp") and vit.get("diastolic_bp")
        else "Not Recorded"
    )
    fv_data = [
        ["FAST Domain", "Manual Observation", "Experimental AI Status", "Vital Sign", "Recorded Value & Provenance"],
        [
            "Facial Weakness (F)",
            str(fast.get("facial_weakness_reported", "N/A")),
            f"{fast.get('facial_ai_status', 'N/A')} (score={fast.get('facial_ai_score', 'N/A')})",
            "Blood Pressure",
            f"{bp_str} [{vit.get('data_source_label', 'N/A')}]",
        ],
        [
            "Arm Weakness (A)",
            str(fast.get("arm_weakness_reported", "N/A")),
            str(fast.get("arm_ai_drift_detected", "N/A")),
            "Heart Rate",
            f"{vit.get('heart_rate') or 'Not Recorded'} bpm",
        ],
        [
            "Speech Difficulty (S)",
            str(fast.get("speech_difficulty_reported", "N/A")),
            "See Acoustic Module",
            "SpO2 / Temp",
            f"SpO2: {vit.get('spo2') or 'N/A'}% | Temp: {vit.get('temperature_c') or 'N/A'} C",
        ],
    ]
    fv_table = Table(fv_data, colWidths=[32 * mm, 28 * mm, 46 * mm, 28 * mm, 44 * mm])
    fv_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#101935")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("PADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(fv_table)

    # 6. Brain CT Findings
    story.append(Paragraph("3. AI Brain CT Scan Analysis (Research Decision Support — Not a Diagnosis)", sec_style))
    if ct and ct.get("file_name"):
        preds = json.loads(ct.get("haemorrhage_pred_json", "{}"))
        pred_rows = [["Haemorrhage Subtype", "Model Score", "Interpretation Note"]]
        for k, v in preds.items():
            pred_rows.append(
                [
                    k.replace("_", " ").title(),
                    f"{float(v):.1%}",
                    "Suspected Flag (>=50%)" if float(v) >= 0.5 else "Below 50% Threshold",
                ]
            )
        pred_rows.append(
            [
                "Model Uncertainty (Entropy)",
                str(ct.get("uncertainty_entropy", "N/A")),
                f"Weights Status: {ct.get('model_weights_status', 'N/A')}",
            ]
        )
        pred_rows.append(
            [
                "Experimental Ischaemic Burden",
                f"{ct.get('ischaemic_lesion_burden_pct', 0.0)}%",
                f"Status: {ct.get('ischaemic_seg_status', 'N/A')}",
            ]
        )
        ct_table = Table(pred_rows, colWidths=[58 * mm, 35 * mm, 85 * mm])
        ct_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1C2541")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(ct_table)

        if ct_visuals and all(
            k in ct_visuals for k in ("brain_window_rgb", "gradcam_overlay_rgb", "segmentation_overlay_rgb")
        ):
            story.append(Spacer(1, 3 * mm))
            img1 = _numpy_rgb_to_rl_image(ct_visuals["brain_window_rgb"])
            img2 = _numpy_rgb_to_rl_image(ct_visuals["gradcam_overlay_rgb"])
            img3 = _numpy_rgb_to_rl_image(ct_visuals["segmentation_overlay_rgb"])
            vis_table = Table(
                [
                    ["Windowed Brain CT", "Grad-CAM Activation Overlay", "Experimental Ischaemic Mask"],
                    [img1, img2, img3],
                ],
                colWidths=[59 * mm, 59 * mm, 59 * mm],
            )
            vis_table.setStyle(
                TableStyle(
                    [
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, 0), 8),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("PADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(vis_table)
    else:
        story.append(
            Paragraph(
                "No Brain CT scan was uploaded for this assessment. Missing imaging never delays symptom-based emergency escalation.",
                body_style,
            )
        )

    # 7. Escalation Reasons, Conflicts & Missing Data
    story.append(Paragraph("4. Multimodal Synthesis, Safety Escalations & Limitations", sec_style))
    reasons = "; ".join(integrated_evaluation.get("escalation_reasons", [])) or "None flagged."
    conflicts = "; ".join(integrated_evaluation.get("conflicts", [])) or "No cross-modal conflicts detected."
    missing = ", ".join(integrated_evaluation.get("missing_information", [])) or "None."
    uncert = "; ".join(integrated_evaluation.get("uncertainty_notes", [])) or "Standard operational parameters."

    synth_rows = [
        ["Reasons for Escalation:", Paragraph(reasons, body_style)],
        ["Cross-Modal Conflicts:", Paragraph(conflicts, body_style)],
        ["Missing Information:", Paragraph(missing, body_style)],
        ["Uncertainty & Limitations:", Paragraph(uncert, body_style)],
    ]
    synth_table = Table(synth_rows, colWidths=[45 * mm, 133 * mm])
    synth_table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(synth_table)

    doc.build(story)
    return output_pdf_path
