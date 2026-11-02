"""
PAGE 1: Main Hospital Dashboard for NeuroScan AI.
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from app.components.ui_cards import render_metric_card, render_top_banner
from config import APP_VERSION, COMPUTE_LABEL
from database.db_manager import DatabaseManager


def render_dashboard_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="Clinical Command Dashboard",
        page_subtitle="Multimodal Stroke Screening Telemetry & AI Module Readiness",
    )

    metrics = db.get_dashboard_metrics()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        render_metric_card(
            title="Total Assessments",
            value=str(metrics["total_assessments"]),
            subtitle=f"Live: {metrics['live_assessments']} | Demo: {metrics['demo_assessments']}",
        )
    with col2:
        render_metric_card(
            title="Urgent Escalations",
            value=str(metrics["urgent_assessments"]),
            subtitle="Triggered FAST or CT Safety Rules",
            is_urgent=metrics["urgent_assessments"] > 0,
        )
    with col3:
        render_metric_card(
            title="Brain CT Scans Analysed",
            value=str(metrics["ct_scans_analysed"]),
            subtitle=f"Suspected ICH Flagged: {metrics['haemorrhage_positive_count']}",
        )
    with col4:
        render_metric_card(
            title="Compute & Runtime",
            value=COMPUTE_LABEL.split(" ")[0],
            subtitle=f"{COMPUTE_LABEL} | v{APP_VERSION}",
        )

    with st.expander("Competition Demonstration Scenario Controls (Fictional Data)", expanded=True):
        st.caption(
            "Seed or purge the 3 mandatory GMU MASE 2026 competition scenarios "
            "(Scenario A: ICH Positive, Scenario B: FAST Positive / CT ICH Negative, "
            "Scenario C: Missing CT / Degraded Sensor Escalation). "
            "All seeded records are explicitly tagged as SIMULATED_DEMO and never mixed with clinical performance."
        )
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            if st.button("Load 3 Competition Demo Scenarios (A, B, C)", use_container_width=True):
                count = db.seed_demo_scenarios()
                st.session_state["active_assessment_id"] = "ASM-DEMOA001"
                st.session_state["active_patient_ref_id"] = "DEMO-SCENARIO-A"
                st.success(f"Loaded {count} fictional demonstration scenarios into SQLite.")
                st.rerun()
        with dcol2:
            if st.button("Purge All Demo Records", use_container_width=True):
                removed = db.delete_demo_records()
                st.info(f"Removed {removed} fictional demonstration patient records.")
                st.rerun()

    st.markdown("### Recent Multimodal Assessment History")
    recent_rows = db.get_recent_assessments(limit=25)

    if not recent_rows:
        st.info(
            "No assessments recorded yet. Click 'Load 3 Competition Demo Scenarios (A, B, C)' above "
            "or open '2. Patient Assessment' to start a new screening."
        )
    else:
        df = pd.DataFrame(recent_rows)
        df["Record Type"] = df["is_demo_record"].apply(
            lambda x: "SIMULATED_DEMO" if x == 1 else "LIVE_ENTRY"
        )
        df["CT Haemorrhage Prob"] = df["overall_haemorrhage_prob"].apply(
            lambda x: f"{x:.1%}" if pd.notnull(x) else "No CT Uploaded"
        )
        display_cols = [
            "assessment_id",
            "patient_ref_id",
            "Record Type",
            "age",
            "symptom_onset_time",
            "urgency_level",
            "facial_weakness_reported",
            "arm_weakness_reported",
            "speech_difficulty_reported",
            "CT Haemorrhage Prob",
            "created_at",
        ]
        st.dataframe(
            df[display_cols].rename(
                columns={
                    "assessment_id": "Assessment ID",
                    "patient_ref_id": "Patient Ref",
                    "age": "Age",
                    "symptom_onset_time": "Onset",
                    "urgency_level": "Urgency Assessment",
                    "facial_weakness_reported": "Face (F)",
                    "arm_weakness_reported": "Arm (A)",
                    "speech_difficulty_reported": "Speech (S)",
                    "created_at": "Timestamp (UTC)",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            urgency_counts = (
                df.groupby(["urgency_level", "Record Type"])
                .size()
                .reset_index(name="Count")
            )
            fig_urg = px.bar(
                urgency_counts,
                x="urgency_level",
                y="Count",
                color="Record Type",
                title="Assessments by Urgency Level & Data Provenance",
                color_discrete_map={
                    "LIVE_ENTRY": "#00B4D8",
                    "SIMULATED_DEMO": "#6366F1",
                },
                template="plotly_dark",
            )
            fig_urg.update_layout(
                paper_bgcolor="#16203D",
                plot_bgcolor="#16203D",
                margin=dict(l=20, r=20, t=45, b=20),
                height=320,
            )
            st.plotly_chart(fig_urg, use_container_width=True)

        with chart_col2:
            fast_summary = pd.DataFrame(
                {
                    "Symptom": ["Facial Weakness", "Arm Weakness", "Speech Difficulty"],
                    "Reported YES": [
                        int((df["facial_weakness_reported"] == "YES").sum()),
                        int((df["arm_weakness_reported"] == "YES").sum()),
                        int((df["speech_difficulty_reported"] == "YES").sum()),
                    ],
                }
            )
            fig_fast = px.bar(
                fast_summary,
                x="Symptom",
                y="Reported YES",
                title="Positive FAST Symptom Distribution Across Cohort",
                color_discrete_sequence=["#EF4444"],
                template="plotly_dark",
            )
            fig_fast.update_layout(
                paper_bgcolor="#16203D",
                plot_bgcolor="#16203D",
                margin=dict(l=20, r=20, t=45, b=20),
                height=320,
            )
            st.plotly_chart(fig_fast, use_container_width=True)

    st.markdown("### AI Model & Safety Subsystem Registry")
    model_meta = db.get_model_metadata()
    meta_df = pd.DataFrame(model_meta)
    meta_df["Checkpoint Ready"] = meta_df["weights_loaded"].apply(
        lambda x: "Ready" if x == 1 else "Untrained Baseline (Ready for .pth)"
    )
    st.dataframe(
        meta_df[
            [
                "model_id",
                "module_name",
                "architecture",
                "dataset_source",
                "Checkpoint Ready",
                "validation_status",
            ]
        ].rename(
            columns={
                "model_id": "Module ID",
                "module_name": "Subsystem",
                "architecture": "Architecture",
                "dataset_source": "Dataset / Standard",
                "validation_status": "Operational Status",
            }
        ),
        use_container_width=True,
        hide_index=True,
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
    render_dashboard_page(_db)
