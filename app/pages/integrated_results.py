"""
PAGE 4: Multimodal Integrated Analysis & Safety Engine Page.
Displays combined findings across every module, provenance tags, conflict alerts,
missing data inventory, and deterministic urgency escalation.
"""
import streamlit as st

from ai_engine.integration_engine import MultimodalIntegrationEngine
from app.components.ui_cards import render_metric_card, render_top_banner
from database.db_manager import DatabaseManager


def render_integrated_results_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="Multimodal Integrated Analysis & Safety Engine",
        page_subtitle="Rule-Based Clinical Synthesis, Cross-Modal Conflict Detection & Provenance Audit",
    )

    recent = db.get_recent_assessments(limit=30)
    if not recent:
        st.info("No assessments available. Load demo scenarios on the Dashboard or create a new Patient Assessment.")
        return

    options = [
        f"{r['assessment_id']} | Patient: {r['patient_ref_id']} ({'DEMO' if r['is_demo_record'] else 'LIVE'})"
        for r in recent
    ]
    active_aid = st.session_state.get("active_assessment_id")
    default_idx = 0
    if active_aid:
        for i, r in enumerate(recent):
            if r["assessment_id"] == active_aid:
                default_idx = i
                break

    selected_label = st.selectbox("Select Assessment Session to Inspect", options=options, index=default_idx)
    chosen_aid = selected_label.split(" | ")[0].strip()
    st.session_state["active_assessment_id"] = chosen_aid

    bundle = db.get_full_assessment_bundle(chosen_aid)
    if not bundle:
        st.error("Assessment record could not be loaded.")
        return

    eval_result = MultimodalIntegrationEngine.evaluate(
        patient_info=bundle.get("patient"),
        fast_results=bundle.get("fast"),
        speech_results=bundle.get("speech"),
        vital_results=bundle.get("vitals"),
        ct_results=bundle.get("ct"),
    )

    urg = eval_result["urgency_level"]
    is_emerg = "EMERGENCY" in urg

    col1, col2, col3 = st.columns(3)
    with col1:
        render_metric_card(
            title="Combined Urgency Level",
            value=urg.replace("_", " "),
            subtitle="Deterministic Safety State Machine",
            is_urgent=is_emerg,
        )
    with col2:
        render_metric_card(
            title="Cross-Modal Conflicts",
            value=str(len(eval_result["conflicts"])),
            subtitle="Discordance Between FAST & Imaging/AI",
            is_urgent=len(eval_result["conflicts"]) > 0,
        )
    with col3:
        render_metric_card(
            title="Missing / Unverified Inputs",
            value=str(len(eval_result["missing_information"])),
            subtitle="Never Assumed Normal When Missing",
        )

    if is_emerg:
        st.error(f"Recommended Next Action: {eval_result['recommended_action']}")
    else:
        st.warning(f"Recommended Next Action: {eval_result['recommended_action']}")

    c_left, c_right = st.columns(2)
    with c_left:
        st.markdown("#### Escalation Triggers & Concerning Indicators")
        if eval_result["escalation_reasons"]:
            for r in eval_result["escalation_reasons"]:
                st.markdown(f"- 🚨 {r}")
        else:
            st.write("No acute escalation triggers flagged.")

        st.markdown("#### Cross-Modal Conflict Alerts")
        if eval_result["conflicts"]:
            for c in eval_result["conflicts"]:
                st.warning(c)
        else:
            st.write("No cross-modal conflicts detected.")

    with c_right:
        st.markdown("#### Missing Information Inventory")
        if eval_result["missing_information"]:
            for m in eval_result["missing_information"]:
                st.markdown(f"- ⚠️ `{m}` (Missing data is never treated as normal)")
        else:
            st.write("All core multimodal screening domains recorded.")

        st.markdown("#### Model Uncertainty & Quality Flags")
        if eval_result["uncertainty_notes"]:
            for u in eval_result["uncertainty_notes"]:
                st.info(u)
        else:
            st.write("No additional sensor degradation flags.")

    st.markdown("### Data Provenance Breakdown (AI vs. User-Reported vs. Vitals vs. Simulated)")
    prov = eval_result["provenance_breakdown"]
    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        st.markdown("**User-Reported Symptoms**")
        for item in prov["USER_REPORTED"] or ["None reported positive"]:
            st.write(f"• {item}")
    with pcol2:
        st.markdown("**AI-Derived Findings**")
        for item in prov["AI_DERIVED"] or ["No AI modules executed"]:
            st.write(f"• {item}")
    with pcol3:
        st.markdown("**Vital Signs & Demo Flags**")
        for item in (prov["VITALS_RECORDED"] + prov["SIMULATED_DEMO_FLAGS"]) or ["No vitals recorded"]:
            st.write(f"• {item}")


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
    render_integrated_results_page(_db)
