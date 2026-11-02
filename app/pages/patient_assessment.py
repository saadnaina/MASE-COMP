"""
PAGE 2: Patient Assessment & FAST Multimodal Assessment Page.
Features TRUE LIVE BROWSER CAPTURE:
- Live Webcam Camera Input (st.camera_input) for Real-Time Smile & Facial Asymmetry
- Live Webcam Camera Input (st.camera_input) for Real-Time Arm Drift / Pose Check
- Live Microphone Audio Recording (st.audio_input) for Real-Time Speech Analysis
- Complete Patient History, Vitals & Automated Safety Integration
"""
import streamlit as st

from ai_engine.arm_processor import ArmWeaknessProcessor
from ai_engine.face_processor import FacialAsymmetryProcessor
from ai_engine.integration_engine import MultimodalIntegrationEngine
from ai_engine.patient_processor import PatientVitalProcessor
from ai_engine.speech_processor import SpeechAnalysisProcessor
from app.components.ui_cards import render_top_banner
from config import SPEECH_REFERENCE_SENTENCE
from database.db_manager import DatabaseManager


@st.cache_resource
def _get_fast_processors():
    return (
        FacialAsymmetryProcessor(),
        ArmWeaknessProcessor(),
        SpeechAnalysisProcessor(),
    )


def render_patient_assessment_page(db: DatabaseManager) -> None:
    render_top_banner(
        page_title="Live Multimodal FAST Assessment",
        page_subtitle="Real-Time Browser Camera & Microphone Screening (No File Uploads Needed)",
    )

    face_proc, arm_proc, speech_proc = _get_fast_processors()

    tab_patient, tab_face, tab_arm, tab_speech = st.tabs(
        [
            "1. Patient Info, Symptoms & Vitals",
            "2. Live Webcam Smile & Facial Test",
            "3. Live Webcam Arm Drift Test",
            "4. Live Microphone Speech Recording",
        ]
    )

    # TAB 2: LIVE WEBCAM SMILE & FACIAL TEST
    with tab_face:
        st.markdown("#### Live Webcam Facial Asymmetry & Smile Screener")
        st.info(
            "**Instructions:** Position the patient centrally in front of the webcam, ask them to give a wide smile, "
            "and click **'Take Live Photo'** below. MediaPipe Face Mesh will instantly measure labial-canthal asymmetry."
        )
        consent = st.checkbox("Patient/User consents to live browser camera frame capture", value=True, key="face_consent")
        if consent:
            live_cam_face = st.camera_input("Capture Live Smile Frame")
            if live_cam_face is not None:
                face_bytes = live_cam_face.getvalue()
                res_face = face_proc.analyse_image_bytes(face_bytes)
                st.session_state["last_face_ai_result"] = res_face
                fcol1, fcol2 = st.columns([1, 1])
                with fcol1:
                    if res_face.get("annotated_rgb") is not None:
                        st.image(
                            res_face["annotated_rgb"],
                            caption=f"Live Facial Landmarks & Symmetry Vector ({res_face['status']})",
                            use_container_width=True,
                        )
                with fcol2:
                    st.write(f"**AI Status:** `{res_face['status']}`")
                    st.write(f"**Labial-Canthal Asymmetry Ratio:** `{res_face.get('asymmetry_score', 'N/A')}`")
                    st.write(f"**Mouth Corner Tilt (deg):** `{res_face.get('mouth_angle_deg', 'N/A')}`")
                    st.write(f"**Mean Frame Luminance:** `{res_face.get('mean_luminance', 'N/A')}`")
                    if res_face.get("asymmetry_flagged"):
                        st.warning(res_face["message"])
                    else:
                        st.success(res_face["message"])

    # TAB 3: LIVE WEBCAM ARM DRIFT TEST
    with tab_arm:
        st.markdown("#### Live Webcam Arm Pronator Drift Assessment")
        st.info(
            "**Instructions:** Ask the patient to extend both arms forward at 90 degrees with palms up and close their eyes "
            "for 10 seconds. Capture a live webcam photo during the hold to measure vertical drift."
        )
        consent_arm = st.checkbox("Patient/User consents to live browser camera frame capture for pose", value=True, key="arm_consent")
        if consent_arm:
            live_cam_arm = st.camera_input("Capture Live Arm Hold Frame")
            if live_cam_arm is not None:
                arm_bytes = live_cam_arm.getvalue()
                res_arm = arm_proc.analyse_pose_image_bytes(arm_bytes)
                st.session_state["last_arm_ai_result"] = res_arm
                acol1, acol2 = st.columns(2)
                with acol1:
                    if res_arm.get("annotated_rgb") is not None:
                        st.image(
                            res_arm["annotated_rgb"],
                            caption="Live Pose Wrist-Shoulder Elevation Overlay",
                            use_container_width=True,
                        )
                with acol2:
                    st.write(f"**Pose AI Status:** `{res_arm['status']}`")
                    st.write(f"**Normalized Wrist Height Delta:** `{res_arm.get('wrist_height_diff_norm', 'N/A')}`")
                    if res_arm.get("drift_flagged"):
                        st.warning(res_arm["message"])
                    else:
                        st.success(res_arm["message"])

    # TAB 4: LIVE MICROPHONE SPEECH RECORDING
    with tab_speech:
        st.markdown("#### Live Microphone Speech & Acoustic Screener")
        st.info(
            f"**Instructions:** Click the microphone record button below and have the patient read aloud this reference sentence:\n"
            f"> *\"{SPEECH_REFERENCE_SENTENCE}\"*"
        )
        
        # Use Streamlit's native st.audio_input for live microphone recording
        try:
            audio_data = st.audio_input("Record Live Speech from Microphone")
        except Exception:
            audio_data = None
            st.warning("`st.audio_input` requires Streamlit >= 1.39.0. Falling back to file uploader for audio.")
            audio_file_fallback = st.file_uploader("Upload Speech Recording (.wav)", type=["wav", "mp3"])
            if audio_file_fallback is not None:
                audio_bytes = audio_file_fallback.getvalue()
            else:
                audio_bytes = None

        if audio_data is not None:
            audio_bytes = audio_data.getvalue()

        if 'audio_bytes' in locals() and audio_bytes is not None:
            manual_trans = st.text_input(
                "Optional Manual Transcription Verification:",
                value="",
            )
            res_speech = speech_proc.analyse_audio_bytes(
                audio_bytes,
                reference_sentence=SPEECH_REFERENCE_SENTENCE,
                manual_transcription_fallback=manual_trans,
            )
            st.session_state["last_speech_ai_result"] = res_speech
            scol1, scol2, scol3, scol4 = st.columns(4)
            scol1.metric("Duration (s)", f"{res_speech['duration_sec']}s")
            scol2.metric("Detected Pauses", str(res_speech["pause_count"]))
            scol3.metric("Speaking Rate", f"{res_speech['speaking_rate_wpm']} WPM")
            scol4.metric("Reference Similarity", f"{res_speech['similarity_score']:.1%}")
            st.write(f"**Audio Quality Flag:** `{res_speech['audio_quality_flag']}`")
            st.write(f"**Transcription:** `{res_speech['transcription']}`")
            st.info(res_speech["message"])

    # TAB 1: MASTER PATIENT & FAST SUBMISSION FORM
    with tab_patient:
        st.markdown("#### Complete Patient History, Manual FAST Observations & Vital Signs")
        with st.form("complete_patient_fast_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                patient_ref = st.text_input(
                    "Patient Reference ID (Pseudonymous)",
                    value=st.session_state.get("active_patient_ref_id", "PT-2026-001"),
                )
                age = st.number_input("Patient Age (Years)", min_value=0, max_value=120, value=64)
                onset = st.text_input("Symptom Onset Time", value="35 minutes ago")
                st.markdown("**Relevant Medical History**")
                hx_htn = st.checkbox("Hypertension", value=True)
                hx_afib = st.checkbox("Atrial Fibrillation")
                hx_dm = st.checkbox("Diabetes Mellitus")
                hx_prior = st.checkbox("Prior Stroke / TIA")
                hx_anticoag = st.checkbox("Anticoagulant Therapy")

            with col2:
                st.markdown("**Manual FAST Observations (Primary Clinical Trigger)**")
                face_rep = st.selectbox("Facial Weakness / Droop Observed?", ["NO", "YES", "UNSURE"])
                arm_rep = st.selectbox("Unilateral Arm Weakness / Drift Observed?", ["NO", "YES", "UNSURE"])
                speech_rep = st.selectbox("Speech Slurring / Word-Finding Difficulty?", ["NO", "YES", "UNSURE"])
                st.caption(
                    "Note: Any 'YES' response triggers immediate emergency escalation regardless of AI or CT results."
                )

            with col3:
                st.markdown("**Vital Signs (Leave 0 if Missing)**")
                vital_source = st.selectbox(
                    "Vital Signs Provenance Label",
                    ["USER_REPORTED", "SENSOR_MEASURED", "SIMULATED_DEMO"],
                )
                sys_bp = st.number_input("Systolic BP (mmHg, 0=Missing)", min_value=0, max_value=300, value=156)
                dia_bp = st.number_input("Diastolic BP (mmHg, 0=Missing)", min_value=0, max_value=200, value=94)
                hr = st.number_input("Heart Rate (bpm, 0=Missing)", min_value=0, max_value=250, value=88)
                spo2 = st.number_input("SpO2 (%, 0=Missing)", min_value=0.0, max_value=100.0, value=96.5)
                temp_c = st.number_input("Temperature (°C, 0=Missing)", min_value=0.0, max_value=45.0, value=37.0)

            submit_btn = st.form_submit_button("Save Patient Assessment & Run Safety Evaluation", use_container_width=True)
            if submit_btn:
                med_hx = {
                    "hypertension": hx_htn,
                    "atrial_fibrillation": hx_afib,
                    "diabetes": hx_dm,
                    "prior_stroke_tia": hx_prior,
                    "anticoagulants": hx_anticoag,
                }
                is_demo = vital_source == "SIMULATED_DEMO"
                clean_id = db.create_or_update_patient(
                    patient_ref_id=patient_ref,
                    age=int(age),
                    symptom_onset_time=onset,
                    medical_history=med_hx,
                    is_demo_record=is_demo,
                )

                face_ai = st.session_state.get("last_face_ai_result", {})
                arm_ai = st.session_state.get("last_arm_ai_result", {})
                speech_ai = st.session_state.get("last_speech_ai_result", {})

                fast_dict = {
                    "facial_weakness_reported": face_rep,
                    "facial_ai_score": face_ai.get("asymmetry_score"),
                    "facial_ai_status": face_ai.get("status", "NOT_PERFORMED"),
                    "arm_weakness_reported": arm_rep,
                    "arm_ai_drift_detected": arm_ai.get("status", "NOT_PERFORMED"),
                    "speech_difficulty_reported": speech_rep,
                }
                vital_dict = {
                    "systolic_bp": int(sys_bp) if sys_bp > 0 else None,
                    "diastolic_bp": int(dia_bp) if dia_bp > 0 else None,
                    "heart_rate": int(hr) if hr > 0 else None,
                    "spo2": float(spo2) if spo2 > 0 else None,
                    "temperature_c": float(temp_c) if temp_c > 0 else None,
                    "data_source_label": vital_source,
                }

                integrated_eval = MultimodalIntegrationEngine.evaluate(
                    patient_info={"patient_ref_id": clean_id, "age": age, "symptom_onset_time": onset},
                    fast_results=fast_dict,
                    speech_results=speech_ai,
                    vital_results=vital_dict,
                    ct_results=None,
                )

                aid = db.create_assessment(
                    patient_ref_id=clean_id,
                    urgency_level=integrated_eval["urgency_level"],
                    status="FAST_COMPLETED",
                    integrated_summary=integrated_eval,
                    is_demo_record=is_demo,
                )
                db.save_vital_signs(
                    assessment_id=aid,
                    systolic_bp=vital_dict["systolic_bp"],
                    diastolic_bp=vital_dict["diastolic_bp"],
                    heart_rate=vital_dict["heart_rate"],
                    spo2=vital_dict["spo2"],
                    temperature_c=vital_dict["temperature_c"],
                    data_source_label=vital_source,
                )
                db.save_fast_assessment(
                    assessment_id=aid,
                    facial_weakness_reported=face_rep,
                    facial_ai_score=face_ai.get("asymmetry_score"),
                    facial_ai_status=face_ai.get("status", "NOT_PERFORMED"),
                    facial_metrics=face_ai if isinstance(face_ai, dict) else {},
                    arm_weakness_reported=arm_rep,
                    arm_ai_drift_detected=arm_ai.get("status", "NOT_PERFORMED"),
                    arm_metrics=arm_ai if isinstance(arm_ai, dict) else {},
                    speech_difficulty_reported=speech_rep,
                )
                if speech_ai:
                    db.save_speech_analysis(
                        assessment_id=aid,
                        transcription=speech_ai.get("transcription", ""),
                        reference_sentence=SPEECH_REFERENCE_SENTENCE,
                        duration_sec=float(speech_ai.get("duration_sec", 0.0)),
                        pause_count=int(speech_ai.get("pause_count", 0)),
                        speaking_rate_wpm=float(speech_ai.get("speaking_rate_wpm", 0.0)),
                        similarity_score=float(speech_ai.get("similarity_score", 0.0)),
                        audio_quality_flag=speech_ai.get("audio_quality_flag", "NOT_PERFORMED"),
                        manual_override_status=speech_rep,
                        metrics=speech_ai,
                    )

                st.session_state["active_assessment_id"] = aid
                st.session_state["active_patient_ref_id"] = clean_id
                st.success(f"Saved Assessment `{aid}` for Patient `{clean_id}`.")
                if integrated_eval["urgency_level"] == "IMMEDIATE_EMERGENCY_ESCALATION":
                    st.error(f"**{integrated_eval['urgency_level']}**: {integrated_eval['recommended_action']}")
                else:
                    st.info(f"**{integrated_eval['urgency_level']}**: {integrated_eval['recommended_action']}")


# Allow direct execution when clicked from Streamlit's native sidebar router
if __name__ == "__main__":
    import sys
    from pathlib import Path
    ROOT_DIR = Path(__file__).resolve().parent.parent.parent
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    from app.styles import apply_medical_theme
    st.set_page_config(page_title="NeuroScan AI | Live Assessment", page_icon="🧠", layout="wide")
    apply_medical_theme()
    for k, v in {"active_assessment_id": "ASM-DEMOA001", "active_patient_ref_id": "DEMO-SCENARIO-A", "current_page": "2. Patient Assessment (FAST & Vitals)"}.items():
        if k not in st.session_state:
            st.session_state[k] = v
    _db = DatabaseManager()
    render_patient_assessment_page(_db)
