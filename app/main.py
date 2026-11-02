"""
Main Streamlit Entrypoint for NeuroScan AI (Unified Super-Version for GMU MASE 2026).
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from app.pages.ct_analysis import render_ct_analysis_page
from app.pages.dashboard import render_dashboard_page
from app.pages.integrated_results import render_integrated_results_page
from app.pages.neuro_simulators import render_neuro_simulators_page
from app.pages.patient_assessment import render_patient_assessment_page
from app.pages.reports import render_reports_page
from app.styles import apply_medical_theme
from config import APP_NAME, APP_VERSION, COMPETITION_NAME, COMPUTE_LABEL
from database.db_manager import DatabaseManager


def init_session_state() -> None:
    defaults = {
        "active_assessment_id": "ASM-DEMOA001",
        "active_patient_ref_id": "DEMO-SCENARIO-A",
        "current_page": "1. Main Dashboard",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def main() -> None:
    st.set_page_config(
        page_title=f"{APP_NAME} | {COMPETITION_NAME}",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_medical_theme()
    init_session_state()

    db = DatabaseManager()

    with st.sidebar:
        st.markdown(f"## 🧠 {APP_NAME}")
        st.caption(f"{COMPETITION_NAME} | Build: `{APP_VERSION}`")
        st.markdown("---")

        nav_options = [
            "1. Main Dashboard",
            "2. Patient Assessment (FAST & Vitals)",
            "3. Brain CT Analysis (ICH & Ischaemic)",
            "4. Integrated Multimodal Analysis",
            "5. Reports & Assessment History",
            "6. Interactive Stroke Simulators & UAE EMS",
        ]
        selected_page = st.radio(
            "Clinical & Interactive Navigation",
            options=nav_options,
            index=nav_options.index(st.session_state["current_page"])
            if st.session_state["current_page"] in nav_options
            else 0,
        )
        st.session_state["current_page"] = selected_page

        st.markdown("---")
        st.markdown("### Active Session Context")
        st.write(f"**Patient Ref:** `{st.session_state['active_patient_ref_id']}`")
        st.write(
            f"**Assessment ID:** `{st.session_state['active_assessment_id'] or 'None Selected'}`"
        )
        st.write(f"**Compute Engine:** `{COMPUTE_LABEL}`")

        st.markdown("---")
        st.warning(
            "**UAE Emergency Protocol (Dial 998):** Sudden facial droop, arm drift, or speech "
            "impairment requires immediate emergency dispatch. Never wait for CT upload or AI output."
        )

    if selected_page == "1. Main Dashboard":
        render_dashboard_page(db)
    elif selected_page == "2. Patient Assessment (FAST & Vitals)":
        render_patient_assessment_page(db)
    elif selected_page == "3. Brain CT Analysis (ICH & Ischaemic)":
        render_ct_analysis_page(db)
    elif selected_page == "4. Integrated Multimodal Analysis":
        render_integrated_results_page(db)
    elif selected_page == "5. Reports & Assessment History":
        render_reports_page(db)
    elif selected_page == "6. Interactive Stroke Simulators & UAE EMS":
        render_neuro_simulators_page(db)


if __name__ == "__main__":
    main()
