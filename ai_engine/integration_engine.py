"""
MODULE SIX: Multimodal Clinical Safety Integration Engine.
Transparent, deterministic rule-based safety state machine that synthesizes:
- Brain CT haemorrhage & ischaemic segmentation findings
- Facial asymmetry (manual & AI)
- Arm weakness/drift (manual & AI)
- Speech impairment (manual & acoustic)
- Patient symptoms, onset window, and vital signs

CRITICAL SAFETY INVARIANTS ENFORCED:
1. Any sudden stroke-like symptom (F-A-S-T) triggers IMMEDIATE_EMERGENCY_ESCALATION.
2. A negative CT haemorrhage result NEVER cancels a concerning FAST assessment.
3. A negative FAST result NEVER rules out stroke.
4. Missing CT scans or missing vitals never become 'normal' results and never block triage.
5. Conflicting findings (e.g., FAST positive + CT haemorrhage negative) are explicitly highlighted.
6. Never outputs an unsupported percentage probability of having a stroke or a confirmed diagnosis.
"""
from typing import Any, Dict, List, Optional


class MultimodalIntegrationEngine:
    """Deterministic safety-aware clinical decision-support synthesizer."""

    @staticmethod
    def evaluate(
        patient_info: Optional[Dict[str, Any]] = None,
        fast_results: Optional[Dict[str, Any]] = None,
        speech_results: Optional[Dict[str, Any]] = None,
        vital_results: Optional[Dict[str, Any]] = None,
        ct_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        patient_info = patient_info or {}
        fast_results = fast_results or {}
        speech_results = speech_results or {}
        vital_results = vital_results or {}
        ct_results = ct_results or {}

        escalation_reasons: List[str] = []
        conflicts: List[str] = []
        missing_info: List[str] = []
        uncertainty_notes: List[str] = []

        # Categorized provenance ledger
        ai_findings: List[str] = []
        user_reported_symptoms: List[str] = []
        measured_vitals_summary: List[str] = []
        simulated_flags: List[str] = []

        # 1. Evaluate User-Reported FAST Symptoms
        face_rep = str(fast_results.get("facial_weakness_reported", "UNKNOWN")).upper()
        arm_rep = str(fast_results.get("arm_weakness_reported", "UNKNOWN")).upper()
        speech_rep = str(fast_results.get("speech_difficulty_reported", "UNKNOWN")).upper()

        fast_user_positive = False
        if face_rep == "YES":
            fast_user_positive = True
            user_reported_symptoms.append("Facial weakness / droop reported (USER_REPORTED)")
            escalation_reasons.append("Sudden facial weakness reported during FAST screening.")
        elif face_rep in ("UNKNOWN", "UNSURE", ""):
            missing_info.append("Definitive Manual Facial Assessment")

        if arm_rep == "YES":
            fast_user_positive = True
            user_reported_symptoms.append("Unilateral arm weakness / drift reported (USER_REPORTED)")
            escalation_reasons.append("Sudden arm weakness/drift reported during FAST screening.")
        elif arm_rep in ("UNKNOWN", "UNSURE", ""):
            missing_info.append("Definitive Manual Arm Weakness Assessment")

        if speech_rep == "YES":
            fast_user_positive = True
            user_reported_symptoms.append("Acute speech difficulty / slurring reported (USER_REPORTED)")
            escalation_reasons.append("Sudden speech impairment reported during FAST screening.")
        elif speech_rep in ("UNKNOWN", "UNSURE", ""):
            missing_info.append("Definitive Manual Speech Assessment")

        # 2. Evaluate Experimental AI Vision & Speech Screening Modules
        face_ai_status = str(fast_results.get("facial_ai_status", "NOT_PERFORMED"))
        face_ai_score = fast_results.get("facial_ai_score")
        if face_ai_status == "ASYMMETRY_FLAGGED":
            ai_findings.append(
                f"Facial Landmark AI flagged asymmetry (ratio={face_ai_score}) [AI_DERIVED - EXPERIMENTAL]"
            )
            escalation_reasons.append("Experimental AI facial landmark screening flagged labial-canthal asymmetry.")
            if face_rep == "NO":
                conflicts.append(
                    "Facial AI flagged geometric asymmetry while manual observation recorded NO facial weakness."
                )
        elif face_ai_status in ("POOR_LIGHTING_REJECTED", "NO_FACE_DETECTED", "MULTIPLE_FACES_REJECTED", "FACE_OUTSIDE_FRAME"):
            uncertainty_notes.append(f"Webcam Facial AI unreliable/rejected ({face_ai_status}); relying on manual clinical check.")
            missing_info.append(f"Valid Webcam Facial Frame ({face_ai_status})")
        elif face_ai_status == "NOT_PERFORMED":
            missing_info.append("AI Facial Landmark Screening")

        arm_ai_status = str(fast_results.get("arm_ai_drift_detected", "NOT_PERFORMED"))
        if arm_ai_status == "DRIFT_FLAGGED":
            ai_findings.append("Pose AI flagged asymmetric wrist elevation [AI_DERIVED - EXPERIMENTAL]")
            escalation_reasons.append("Experimental AI pose tracking flagged unilateral arm drift.")
            if arm_rep == "NO":
                conflicts.append(
                    "Pose AI flagged wrist height asymmetry while manual observation recorded NO arm weakness."
                )
        elif arm_ai_status == "NOT_PERFORMED":
            missing_info.append("AI Pose Arm Drift Screening")

        # Speech module check
        if speech_results:
            qual = speech_results.get("audio_quality_flag", "NOT_PERFORMED")
            sim = speech_results.get("similarity_score")
            if qual == "VALID_AUDIO" and sim is not None:
                ai_findings.append(
                    f"Speech acoustic check completed (reference similarity={sim:.2f}, "
                    f"rate={speech_results.get('speaking_rate_wpm', 0)} WPM) [AI_DERIVED - EXPERIMENTAL]"
                )
            elif qual in ("LOW_VOLUME_NOISY", "TOO_SHORT_UNRELIABLE", "INVALID_AUDIO_FILE"):
                uncertainty_notes.append(f"Audio recording quality flagged as {qual}.")
        else:
            missing_info.append("Acoustic Speech Recording Analysis")

        # 3. Evaluate Vital Signs & Provenance
        vital_source = str(vital_results.get("data_source_label", "MISSING"))
        sys_bp = vital_results.get("systolic_bp")
        dia_bp = vital_results.get("diastolic_bp")
        hr = vital_results.get("heart_rate")
        spo2 = vital_results.get("spo2")

        if vital_source == "SIMULATED_DEMO":
            simulated_flags.append("Vital signs are marked as SIMULATED_DEMO for competition demonstration.")

        if sys_bp is not None and dia_bp is not None:
            measured_vitals_summary.append(f"BP: {sys_bp}/{dia_bp} mmHg [{vital_source}]")
            if sys_bp >= 180 or dia_bp >= 110:
                escalation_reasons.append(
                    f"Severely elevated blood pressure ({sys_bp}/{dia_bp} mmHg) requiring urgent clinical evaluation."
                )
        else:
            missing_info.append("Blood Pressure (Systolic/Diastolic)")

        if hr is not None:
            measured_vitals_summary.append(f"HR: {hr} bpm [{vital_source}]")
        else:
            missing_info.append("Heart Rate")

        if spo2 is not None:
            measured_vitals_summary.append(f"SpO2: {spo2}% [{vital_source}]")
            if spo2 < 94.0:
                escalation_reasons.append(f"Low oxygen saturation (SpO2 {spo2}%) requiring urgent support.")
        else:
            missing_info.append("Oxygen Saturation (SpO2)")

        # 4. Evaluate Brain CT Findings & Cross-Modal Safety Conflicts
        ct_present = bool(ct_results and ct_results.get("file_name"))
        ct_ich_detected = False
        if not ct_present:
            missing_info.append("Non-Contrast Brain CT Scan")
            uncertainty_notes.append(
                "No Brain CT scan uploaded. Symptom triage proceeds immediately without waiting for imaging."
            )
        else:
            img_qual = str(ct_results.get("image_quality_status", "VALID"))
            if img_qual != "VALID":
                uncertainty_notes.append(f"Uploaded CT image flagged as {img_qual}; interpret with extreme caution.")

            ich_prob = float(ct_results.get("overall_haemorrhage_prob", 0.0) or 0.0)
            ct_ich_detected = bool(ct_results.get("haemorrhage_detected", False) or ich_prob >= 0.50)
            entropy = ct_results.get("uncertainty_entropy")
            weights_stat = str(ct_results.get("model_weights_status", "UNKNOWN"))

            ai_findings.append(
                f"CT Haemorrhage Classifier overall score: {ich_prob:.1%} "
                f"(Entropy={entropy}, Status={weights_stat}) [AI_DERIVED]"
            )

            if "UNTRAINED" in weights_stat.upper():
                uncertainty_notes.append(
                    "CT model ran with untrained baseline architecture weights; probabilities are for pipeline verification only."
                )

            if ct_ich_detected:
                escalation_reasons.append(
                    f"AI Brain CT classifier flagged suspected intracranial haemorrhage (score {ich_prob:.1%})."
                )
                if not fast_user_positive and face_rep == "NO" and arm_rep == "NO" and speech_rep == "NO":
                    conflicts.append(
                        "CT classifier flagged suspected haemorrhage despite negative user-reported FAST symptoms."
                    )
            else:
                # CRITICAL SAFETY INVARIANT: Negative CT Haemorrhage + Positive FAST
                if fast_user_positive or face_ai_status == "ASYMMETRY_FLAGGED" or arm_ai_status == "DRIFT_FLAGGED":
                    conflicts.append(
                        "CRITICAL CROSS-MODAL DISCORDANCE: Clinical FAST screening is POSITIVE while CT Haemorrhage "
                        "classification is NEGATIVE. A negative haemorrhage CT NEVER rules out acute ischaemic stroke."
                    )
                    escalation_reasons.append(
                        "MANDATORY SAFETY RULE: Positive stroke symptoms with negative CT haemorrhage require immediate "
                        "emergency evaluation for suspected acute ischaemic stroke."
                    )

            isc_burden = ct_results.get("ischaemic_lesion_burden_pct")
            isc_status = ct_results.get("ischaemic_seg_status", "NOT_PERFORMED")
            if isc_burden is not None and isc_status != "NOT_PERFORMED":
                ai_findings.append(
                    f"Experimental CT Ischaemic Segmentation burden: {isc_burden}% ({isc_status}) [AI_DERIVED - EXPERIMENTAL]"
                )

        # 5. Determine Final Urgency Tier (Rule-Based State Machine)
        if fast_user_positive or ct_ich_detected:
            urgency_level = "IMMEDIATE_EMERGENCY_ESCALATION"
            recommended_action = (
                "IMMEDIATE EMERGENCY ESCALATION: Activate emergency stroke protocol (Call Emergency Services / Stroke Team) "
                "immediately. Do not delay care for additional uploads. Note: Negative CT haemorrhage never excludes ischaemic stroke."
            )
        elif (
            face_ai_status == "ASYMMETRY_FLAGGED"
            or arm_ai_status == "DRIFT_FLAGGED"
            or len(escalation_reasons) > 0
            or "UNSURE" in (face_rep, arm_rep, speech_rep)
        ):
            urgency_level = "HIGH_PRIORITY_CLINICAL_REVIEW"
            recommended_action = (
                "HIGH PRIORITY CLINICAL REVIEW: Prompt in-person neurological examination and urgent diagnostic workup "
                "recommended due to equivocal symptoms, experimental AI flags, or abnormal vital signs."
            )
        else:
            urgency_level = "STANDARD_CLINICAL_ASSESSMENT"
            recommended_action = (
                "STANDARD CLINICAL ASSESSMENT: No acute FAST symptoms or CT haemorrhage flags currently identified. "
                "SAFETY WARNING: A negative FAST or AI result NEVER rules out stroke (e.g., posterior circulation stroke). "
                "Seek immediate medical care if any neurological symptoms are present or evolving."
            )

        return {
            "urgency_level": urgency_level,
            "recommended_action": recommended_action,
            "escalation_reasons": escalation_reasons,
            "conflicts": conflicts,
            "missing_information": missing_info,
            "uncertainty_notes": uncertainty_notes,
            "provenance_breakdown": {
                "AI_DERIVED": ai_findings,
                "USER_REPORTED": user_reported_symptoms,
                "VITALS_RECORDED": measured_vitals_summary,
                "SIMULATED_DEMO_FLAGS": simulated_flags,
            },
            "non_diagnostic_statement": (
                "NeuroScan AI does NOT calculate an unsupported probability of stroke and never generates "
                "a confirmed clinical diagnosis."
            ),
        }
