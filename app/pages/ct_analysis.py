"""
PAGE 3: AI Brain CT Scan Analyser Page.
Supports:
- DICOM (.dcm) and Raster (.png, .jpg) single-slice or volumetric multi-slice upload
- Built-in Synthetic DICOM Phantom Generator for immediate offline testing
- Interactive Slice Navigation & Windowing Inspection
- DenseNet121 5-Subtype Haemorrhage Classification + Predictive Entropy
- Authentic PyTorch Grad-CAM Visualisation
- Experimental MONAI 2D U-Net Ischaemic Lesion Segmentation Overlay
"""
import json
import pandas as pd
import plotly.express as px
import streamlit as st

from ai_engine.ct_processor import CTHaemorrhageProcessor
from ai_engine.integration_engine import MultimodalIntegrationEngine
from ai_engine.ischaemic_processor import IschaemicSegmentationProcessor
from app.components.ui_cards import render_top_banner
from config import DEMO_CT_DIR, ICH_SUBTYPES
from database.db_manager import DatabaseManager
from preprocessing.dicom_utils import (
    InvalidMedicalImageError,
    generate_synthetic_demo_dicom,
    load_ct_slice_or_volume,
)


@st.cache_resource
def _load_ct_models():
    return CTHaemorrhageProcessor(), IschaemicSegmentationProcessor()


def render_ct_analysis_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="AI Brain CT Scan Analyser",
        page_subtitle="Non-Contrast CT Intracranial Haemorrhage Classification, Grad-CAM & Experimental Ischaemic Segmentation",
    )

    ich_proc, isc_proc = _load_ct_models()

    # Model Readiness Banner
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        st.caption(
            f"**Haemorrhage Classifier (DenseNet121):** `{ich_proc.model_status}` "
            "| Multi-Window HU Input: Brain (80/40), Subdural (200/80), Bone (380/40)"
        )
    with mcol2:
        st.caption(
            f"**Ischaemic Segmenter (MONAI 2D U-Net):** `{isc_proc.status_label}` "
            "| Separate experimental segmentation pipeline (AISD/CPAISD architecture)"
        )

    # Demo Phantom Generator for instant offline competition verification
    with st.expander("Offline DICOM Phantom Generator (Create Test DICOM Slices)", expanded=False):
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            if st.button("Generate & Load Synthetic ICH-Positive DICOM Phantom", use_container_width=True):
                dcm_path = generate_synthetic_demo_dicom(
                    DEMO_CT_DIR / "phantom_haemorrhage_positive.dcm", "haemorrhage_positive"
                )
                st.session_state["preloaded_dicom_bytes"] = [(dcm_path.name, dcm_path.read_bytes())]
                st.success(f"Generated calibrated DICOM phantom: {dcm_path.name}")
        with pcol2:
            if st.button("Generate & Load Synthetic Ischaemic Hypodense DICOM Phantom", use_container_width=True):
                dcm_path = generate_synthetic_demo_dicom(
                    DEMO_CT_DIR / "phantom_ischaemic_hypodense.dcm", "ischaemic_hypodense"
                )
                st.session_state["preloaded_dicom_bytes"] = [(dcm_path.name, dcm_path.read_bytes())]
                st.success(f"Generated calibrated DICOM phantom: {dcm_path.name}")

    uploaded = st.file_uploader(
        "Upload Non-Contrast Brain CT Scan (.dcm, .dicom, .png, .jpg) — Supports Multi-Slice Volumetric Studies",
        type=["dcm", "dicom", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    file_tuples = []
    if uploaded:
        file_tuples = [(f.name, f.getvalue()) for f in uploaded]
    elif "preloaded_dicom_bytes" in st.session_state:
        file_tuples = st.session_state["preloaded_dicom_bytes"]

    if not file_tuples:
        st.info(
            "Upload one or more `.dcm` or `.png/.jpg` CT slices above, or expand "
            "**Offline DICOM Phantom Generator** to generate a calibrated DICOM slice immediately."
        )
        return

    try:
        study = load_ct_slice_or_volume(file_tuples)
    except InvalidMedicalImageError as exc:
        st.error(f"Rejected Invalid/Unsupported Medical Image: {exc}")
        return

    num_slices = study["num_slices"]
    slice_idx = 0
    if num_slices > 1:
        slice_idx = st.slider(
            "Volumetric Slice Navigator",
            min_value=0,
            max_value=num_slices - 1,
            value=num_slices // 2,
        )

    active_hu_slice = study["slices_hu"][slice_idx]
    active_slice_name = study["slice_names"][slice_idx]

    st.write(
        f"**Active Slice:** `{active_slice_name}` | **Format:** `{study['modality_format']}` | "
        f"**Image Quality Check:** `{study['image_quality_status']}` | **Dimensions:** `{active_hu_slice.shape}`"
    )

    target_sub_name = st.selectbox(
        "Grad-CAM Target Subtype Layer",
        options=ICH_SUBTYPES,
        index=len(ICH_SUBTYPES) - 1,
    )
    target_idx = ICH_SUBTYPES.index(target_sub_name)

    if st.button("Run AI Brain CT Analysis (DenseNet121 + Grad-CAM + Ischaemic U-Net)", type="primary", use_container_width=True):
        with st.spinner("Executing multi-window HU conversion, DenseNet121 Grad-CAM, and MONAI U-Net segmentation..."):
            ich_res = ich_proc.analyse_slice(active_hu_slice, target_class_idx=target_idx)
            isc_res = isc_proc.segment_slice(active_hu_slice)

            st.session_state["last_ct_analysis"] = {
                "file_name": active_slice_name,
                "modality_format": study["modality_format"],
                "image_quality_status": study["image_quality_status"],
                "ich": ich_res,
                "isc": isc_res,
            }
            st.session_state["last_ct_visuals"] = {
                "brain_window_rgb": ich_res["brain_window_rgb"],
                "gradcam_heatmap_rgb": ich_res["gradcam_heatmap_rgb"],
                "gradcam_overlay_rgb": ich_res["gradcam_overlay_rgb"],
                "segmentation_overlay_rgb": isc_res["segmentation_overlay_rgb"],
            }

            # Persist to active assessment, ensuring patient and assessment rows exist in SQLite
            pat_id = st.session_state.get("active_patient_ref_id", "DEMO-SCENARIO-A")
            db.create_or_update_patient(pat_id, 64, "40 minutes ago", {}, is_demo_record=False)
            
            aid = st.session_state.get("active_assessment_id")
            if not aid:
                aid = db.create_assessment(pat_id, status="CT_ANALYSED")
                st.session_state["active_assessment_id"] = aid
            else:
                # Ensure the assessment record itself exists in the database table to satisfy foreign keys
                try:
                    db.create_assessment(pat_id, status="CT_ANALYSED", assessment_id=aid)
                except Exception:
                    pass

            db.save_ct_result(
                assessment_id=aid,
                file_name=active_slice_name,
                modality_format=study["modality_format"],
                image_quality_status=study["image_quality_status"],
                haemorrhage_pred=ich_res["subtype_probabilities"],
                overall_haemorrhage_prob=ich_res["overall_haemorrhage_prob"],
                haemorrhage_detected=ich_res["haemorrhage_detected"],
                uncertainty_entropy=ich_res["uncertainty_entropy"],
                ischaemic_seg_status=isc_res["status_label"],
                ischaemic_lesion_burden_pct=isc_res["lesion_burden_pct"],
                model_weights_status=ich_res["model_weights_status"],
                gradcam_available=True,
            )

            # Recompute integrated assessment bundle
            bundle = db.get_full_assessment_bundle(aid)
            if bundle:
                eval_out = MultimodalIntegrationEngine.evaluate(
                    patient_info=bundle.get("patient"),
                    fast_results=bundle.get("fast"),
                    speech_results=bundle.get("speech"),
                    vital_results=bundle.get("vitals"),
                    ct_results=bundle.get("ct"),
                )
                db.create_assessment(
                    patient_ref_id=bundle["assessment"]["patient_ref_id"],
                    urgency_level=eval_out["urgency_level"],
                    status="INTEGRATED_COMPLETED",
                    integrated_summary=eval_out,
                    is_demo_record=bool(bundle["assessment"]["is_demo_record"]),
                    assessment_id=aid,
                )

    if "last_ct_analysis" in st.session_state:
        res = st.session_state["last_ct_analysis"]
        ich = res["ich"]
        isc = res["isc"]

        st.markdown("### Visual Explanations (Grad-CAM & Experimental Segmentation)")
        vcol1, vcol2, vcol3, vcol4 = st.columns(4)
        with vcol1:
            st.image(ich["brain_window_rgb"], caption="1. Windowed CT (W=80, L=40)", use_container_width=True)
        with vcol2:
            st.image(
                ich["gradcam_heatmap_rgb"],
                caption=f"2. Grad-CAM Map ({ich['target_gradcam_subtype']})",
                use_container_width=True,
            )
        with vcol3:
            st.image(ich["gradcam_overlay_rgb"], caption="3. Grad-CAM CT Overlay", use_container_width=True)
        with vcol4:
            st.image(
                isc["segmentation_overlay_rgb"],
                caption=f"4. Experimental Ischaemic Mask ({isc['lesion_burden_pct']}%)",
                use_container_width=True,
            )

        st.markdown("### Subtype Prediction Scores & Model Uncertainty")
        kcol1, kcol2, kcol3 = st.columns(3)
        kcol1.metric("Overall Haemorrhage Score", f"{ich['overall_haemorrhage_prob']:.1%}")
        kcol2.metric("Predictive Uncertainty (Entropy)", f"{ich['uncertainty_entropy']:.3f} bits")
        kcol3.metric("Experimental Ischaemic Burden", f"{isc['lesion_burden_pct']:.2f}%")

        if not ich["weights_loaded"]:
            st.warning(
                "**Untrained Baseline Weights Notice:** No external `.pth` checkpoint was found in "
                "`models/ct_haemorrhage/densenet121_rsna_ich.pth`. The scores and Grad-CAM maps above are computed "
                "genuinely from the initialized DenseNet121 network to verify pipeline mechanics, NOT clinical accuracy."
            )

        sub_df = pd.DataFrame(
            [
                {"Subtype": k.replace("_", " ").title(), "Score": v}
                for k, v in ich["subtype_probabilities"].items()
            ]
        )
        fig = px.bar(
            sub_df,
            x="Score",
            y="Subtype",
            orientation="h",
            range_x=[0.0, 1.0],
            title="Haemorrhage Subtype Output Scores (Multi-Label DenseNet121)",
            color="Score",
            color_continuous_scale="Tealgrn",
            template="plotly_dark",
        )
        fig.update_layout(paper_bgcolor="#16203D", plot_bgcolor="#16203D", height=300)
        st.plotly_chart(fig, use_container_width=True)

        export_json = json.dumps(
            {
                "file_name": res["file_name"],
                "modality_format": res["modality_format"],
                "image_quality_status": res["image_quality_status"],
                "haemorrhage_probabilities": ich["subtype_probabilities"],
                "uncertainty_entropy": ich["uncertainty_entropy"],
                "model_weights_status": ich["model_weights_status"],
                "ischaemic_segmentation_status": isc["status_label"],
                "ischaemic_lesion_burden_pct": isc["lesion_burden_pct"],
                "medical_limitation": "Experimental output. Negative haemorrhage score NEVER excludes ischaemic stroke.",
            },
            indent=2,
        )
        st.download_button(
            "Download CT Analysis JSON Summary",
            data=export_json,
            file_name=f"ct_analysis_{res['file_name']}.json",
            mime="application/json",
        )


# Allow direct execution when clicked from Streamlit's native sidebar router (e.g., /dashboard)
if __name__ == "__main__":
    import sys
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from app.styles import apply_medical_theme
    st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="wide")
    apply_medical_theme()
    for k, v in {"active_assessment_id": "ASM-DEMOA001", "active_patient_ref_id": "DEMO-SCENARIO-A", "current_page": "1. Main Dashboard"}.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _db = DatabaseManager()
    render_ct_analysis_page(_db)
