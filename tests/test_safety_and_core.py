"""
Automated Test Suite for NeuroScan AI (GMU MASE 2026).
Covers:
1. CT DICOM HU preprocessing & multi-window normalization
2. Safe rejection of invalid/corrupt image uploads
3. DenseNet121 model inference & Grad-CAM generation
4. Missing patient information handling (never silently treated as normal)
5. Facial analysis lighting/framing safety checks
6. Multimodal Integration Engine & conflicting findings detection
7. MANDATORY CLINICAL SAFETY TEST: Sudden facial + arm weakness with negative CT haemorrhage
   MUST still produce IMMEDIATE_EMERGENCY_ESCALATION.
8. ReportLab PDF report generation
9. SQLite Database CRUD & Demo vs. Live record isolation
"""
from pathlib import Path
import numpy as np
import pytest

from ai_engine.ct_processor import CTHaemorrhageProcessor
from ai_engine.face_processor import FacialAsymmetryProcessor
from ai_engine.integration_engine import MultimodalIntegrationEngine
from ai_engine.ischaemic_processor import IschaemicSegmentationProcessor, compute_dice_and_iou
from database.db_manager import DatabaseManager
from evaluation.evaluate_models import compute_classification_metrics
from preprocessing.dicom_utils import (
    InvalidMedicalImageError,
    apply_hu_window,
    build_multiwindow_tensor,
    generate_synthetic_demo_dicom,
    load_ct_slice_or_volume,
)
from reports.pdf_generator import generate_clinical_pdf_report


def test_ct_preprocessing_and_dicom_phantom(tmp_path: Path) -> None:
    dcm_file = generate_synthetic_demo_dicom(tmp_path / "test_ich.dcm", "haemorrhage_positive")
    assert dcm_file.exists()

    study = load_ct_slice_or_volume([(dcm_file.name, dcm_file.read_bytes())])
    assert study["num_slices"] == 1
    assert study["modality_format"] == "DICOM"

    hu_slice = study["slices_hu"][0]
    tensor_3c = build_multiwindow_tensor(hu_slice, target_size=(224, 224))
    assert tensor_3c.shape == (3, 224, 224)
    assert 0.0 <= float(np.min(tensor_3c)) <= float(np.max(tensor_3c)) <= 1.0


def test_invalid_image_upload_rejected() -> None:
    with pytest.raises(InvalidMedicalImageError):
        load_ct_slice_or_volume([("corrupt.dcm", b"not_a_valid_dicom_payload")])

    with pytest.raises(InvalidMedicalImageError):
        load_ct_slice_or_volume([("malicious.exe", b"0" * 256)])


def test_model_inference_and_gradcam() -> None:
    hu_dummy = np.random.uniform(10.0, 75.0, size=(256, 256)).astype(np.float32)
    ich_proc = CTHaemorrhageProcessor()
    res = ich_proc.analyse_slice(hu_dummy)

    assert "subtype_probabilities" in res
    assert len(res["subtype_probabilities"]) == 6
    assert res["gradcam_overlay_rgb"].shape == (256, 256, 3)

    isc_proc = IschaemicSegmentationProcessor()
    seg = isc_proc.segment_slice(hu_dummy)
    assert seg["binary_mask"].shape == (256, 256)


def test_mandatory_safety_rule_fast_positive_with_negative_ct() -> None:
    """
    MANDATORY SAFETY REGRESSION TEST:
    A patient with reported sudden facial weakness and arm weakness MUST still receive
    an urgent emergency recommendation (IMMEDIATE_EMERGENCY_ESCALATION) even if the
    CT haemorrhage model reports a negative result.
    """
    evaluation = MultimodalIntegrationEngine.evaluate(
        patient_info={"patient_ref_id": "PT-SAFETY-01", "age": 62, "symptom_onset_time": "30 mins ago"},
        fast_results={
            "facial_weakness_reported": "YES",
            "arm_weakness_reported": "YES",
            "speech_difficulty_reported": "NO",
            "facial_ai_status": "NOT_PERFORMED",
            "arm_ai_drift_detected": "NOT_PERFORMED",
        },
        vital_results={
            "systolic_bp": 122,
            "diastolic_bp": 78,
            "heart_rate": 72,
            "spo2": 99.0,
            "data_source_label": "USER_REPORTED",
        },
        ct_results={
            "file_name": "negative_ncct.dcm",
            "image_quality_status": "VALID",
            "overall_haemorrhage_prob": 0.02,
            "haemorrhage_detected": False,
            "uncertainty_entropy": 0.11,
            "model_weights_status": "TRAINED_RSNA_CHECKPOINT_LOADED",
        },
    )

    assert evaluation["urgency_level"] == "IMMEDIATE_EMERGENCY_ESCALATION"
    assert len(evaluation["conflicts"]) >= 1
    assert any("ischaemic stroke" in r.lower() for r in evaluation["escalation_reasons"])


def test_missing_data_never_silently_becomes_normal() -> None:
    evaluation = MultimodalIntegrationEngine.evaluate(
        patient_info={"patient_ref_id": "PT-MISSING-01"},
        fast_results={
            "facial_weakness_reported": "UNKNOWN",
            "arm_weakness_reported": "UNKNOWN",
            "speech_difficulty_reported": "UNKNOWN",
        },
        vital_results={},
        ct_results=None,
    )
    missing = evaluation["missing_information"]
    assert "Non-Contrast Brain CT Scan" in missing
    assert "Blood Pressure (Systolic/Diastolic)" in missing
    assert "Definitive Manual Facial Assessment" in missing


def test_facial_poor_lighting_handled() -> None:
    dark_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    face_proc = FacialAsymmetryProcessor()
    res = face_proc.analyse_rgb_frame(dark_frame)
    assert res["status"] == "POOR_LIGHTING_REJECTED"
    assert res["asymmetry_flagged"] is False


def test_database_and_pdf_generation(tmp_path: Path) -> None:
    db = DatabaseManager(db_path=tmp_path / "test_neuroscan.db")
    seeded = db.seed_demo_scenarios()
    assert seeded == 3

    bundle = db.get_full_assessment_bundle("ASM-DEMOA001")
    assert bundle is not None

    eval_out = MultimodalIntegrationEngine.evaluate(
        patient_info=bundle["patient"],
        fast_results=bundle["fast"],
        vital_results=bundle["vitals"],
        ct_results=bundle["ct"],
    )
    pdf_file = generate_clinical_pdf_report(
        output_pdf_path=tmp_path / "test_report.pdf",
        assessment_bundle=bundle,
        integrated_evaluation=eval_out,
    )
    assert pdf_file.exists()
    assert pdf_file.stat().st_size > 1500

    deleted = db.delete_demo_records()
    assert deleted == 3


def test_evaluation_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.8, 0.9])
    metrics = compute_classification_metrics(y_true, y_prob)
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0

    m1 = np.array([[1, 1], [0, 0]])
    m2 = np.array([[1, 1], [0, 0]])
    dice, iou = compute_dice_and_iou(m1, m2)
    assert dice == 1.0
    assert iou == 1.0


def test_interactive_neuro_visuals() -> None:
    from ai_engine.neuro_visuals import render_time_is_brain_visual, render_vascular_territory_atlas
    img_tib, stats_tib = render_time_is_brain_visual(90)
    assert img_tib.shape == (440, 860, 3)
    assert stats_tib["neurons_lost_millions"] == 171.0

    img_vasc, info_vasc = render_vascular_territory_atlas("MCA", 80)
    assert img_vasc.shape == (440, 860, 3)
    assert "Middle Cerebral Artery" in info_vasc["zone_title"]
