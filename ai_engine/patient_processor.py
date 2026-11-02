"""
MODULE FIVE: Patient Information & Vital Signs Validator.
Handles missing value normalization, provenance enforcement, and clinical range flags
while ensuring normal vital signs never rule out stroke.
"""
from typing import Any, Dict, List, Optional


class PatientVitalProcessor:
    """Processes demographic, symptom onset, and vital sign inputs safely."""

    @staticmethod
    def evaluate_vitals_and_history(
        age: Optional[int],
        symptom_onset_time: Optional[str],
        systolic_bp: Optional[int],
        diastolic_bp: Optional[int],
        heart_rate: Optional[int],
        spo2: Optional[float],
        temperature_c: Optional[float],
        data_source_label: str,
        medical_history: Dict[str, bool],
    ) -> Dict[str, Any]:
        missing_fields: List[str] = []
        abnormal_flags: List[str] = []

        if not symptom_onset_time or symptom_onset_time.strip() == "":
            missing_fields.append("Symptom Onset Time")

        if systolic_bp is None or diastolic_bp is None:
            missing_fields.append("Blood Pressure (Systolic/Diastolic)")
        else:
            if systolic_bp >= 180 or diastolic_bp >= 110:
                abnormal_flags.append(
                    f"Severe Hypertension ({systolic_bp}/{diastolic_bp} mmHg) [{data_source_label}]"
                )
            elif systolic_bp < 90:
                abnormal_flags.append(
                    f"Hypotension ({systolic_bp}/{diastolic_bp} mmHg) [{data_source_label}]"
                )

        if heart_rate is None:
            missing_fields.append("Heart Rate")
        elif heart_rate > 110 or heart_rate < 50:
            abnormal_flags.append(f"Abnormal Heart Rate ({heart_rate} bpm) [{data_source_label}]")

        if spo2 is None:
            missing_fields.append("SpO2 Oxygen Saturation")
        elif spo2 < 94.0:
            abnormal_flags.append(f"Hypoxaemia SpO2 ({spo2:.1f}%) [{data_source_label}]")

        if temperature_c is None:
            missing_fields.append("Body Temperature")
        elif temperature_c >= 38.0:
            abnormal_flags.append(f"Pyrexia / Fever ({temperature_c:.1f} °C) [{data_source_label}]")

        active_risk_factors = [k.replace("_", " ").title() for k, v in medical_history.items() if v]

        return {
            "data_source_label": data_source_label,
            "missing_vital_fields": missing_fields,
            "abnormal_vital_flags": abnormal_flags,
            "documented_risk_factors": active_risk_factors,
            "safety_invariant_note": (
                "Normal blood pressure, heart rate, or SpO2 NEVER rules out acute ischaemic or haemorrhagic stroke."
            ),
        }
