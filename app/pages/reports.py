"""
PAGE 5: Reports & Assessment History Page.
Provides searchable/filterable assessment archive and one-click ReportLab PDF generation.
"""
import pandas as pd
import streamlit as st

from ai_engine.integration_engine import MultimodalIntegrationEngine
from app.components.ui_cards import render_top_banner
from config import REPORTS_DIR
from database.db_manager import DatabaseManager
from reports.pdf_generator import generate_clinical_pdf_report


def render_reports_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="Clinical Reports & Assessment Archive",
        page_subtitle="Searchable Multimodal History & Automated ReportLab PDF Generation",
    )

    records = db.get_recent_assessments(limit=100)
    if not records:
        st.info("No assessment records stored yet. Seed the competition scenarios on the Dashboard first.")
        return

    df = pd.DataFrame(records)
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        search_q = st.text_input("Filter by Patient Reference ID or Assessment ID", value="")
    with fcol2:
        urg_filter = st.selectbox(
            "Filter by Urgency Level",
            ["ALL", "IMMEDIATE_EMERGENCY_ESCALATION", "HIGH_PRIORITY_CLINICAL_REVIEW", "STANDARD_CLINICAL_ASSESSMENT"],
        )

    if search_q.strip():
        q = search_q.strip().upper()
        df = df[
            df["patient_ref_id"].str.upper().str.contains(q, na=False)
            | df["assessment_id"].str.upper().str.contains(q, na=False)
        ]
    if urg_filter != "ALL":
        df = df[df["urgency_level"] == urg_filter]

    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Generate Official Clinical Assessment PDF Report (ReportLab)")
    aid_list = df["assessment_id"].tolist()
    if not aid_list:
        st.warning("No matching assessments to export.")
        return

    selected_aid = st.selectbox("Select Assessment ID for PDF Export", options=aid_list)
    if st.button("Generate & Download PDF Assessment Report", type="primary"):
        bundle = db.get_full_assessment_bundle(selected_aid)
        if bundle:
            eval_out = MultimodalIntegrationEngine.evaluate(
                patient_info=bundle.get("patient"),
                fast_results=bundle.get("fast"),
                speech_results=bundle.get("speech"),
                vital_results=bundle.get("vitals"),
                ct_results=bundle.get("ct"),
            )
            pdf_path = REPORTS_DIR / f"NeuroScan_Report_{selected_aid}.pdf"
            ct_vis = st.session_state.get("last_ct_visuals")
            generate_clinical_pdf_report(
                output_pdf_path=pdf_path,
                assessment_bundle=bundle,
                integrated_evaluation=eval_out,
                ct_visuals=ct_vis,
            )
            db.record_generated_report(selected_aid, str(pdf_path))
            pdf_bytes = pdf_path.read_bytes()
            st.success(f"Generated clinical PDF report: `{pdf_path.name}`")
            st.download_button(
                label=f"Download {pdf_path.name}",
                data=pdf_bytes,
                file_name=pdf_path.name,
                mime="application/pdf",
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
    render_reports_page(_db)
