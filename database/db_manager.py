"""
SQLite Database Manager for NeuroScan AI.
Handles schema creation, parameterized CRUD queries, model registry metadata,
and fictional competition demonstration record management.
"""
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import APP_VERSION, DB_PATH, HAEMORRHAGE_WEIGHTS_PATH, ISCHAEMIC_WEIGHTS_PATH


class DatabaseManager:
    """Thread-safe SQLite database interface for clinical screening records."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_schema()
        self._sync_model_metadata()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def _initialize_schema(self) -> None:
        schema_sql = """
        CREATE TABLE IF NOT EXISTS patients (
            patient_ref_id TEXT PRIMARY KEY,
            age INTEGER,
            symptom_onset_time TEXT,
            medical_history_json TEXT NOT NULL DEFAULT '{}',
            is_demo_record INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS assessments (
            assessment_id TEXT PRIMARY KEY,
            patient_ref_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'IN_PROGRESS',
            urgency_level TEXT NOT NULL DEFAULT 'PENDING_EVALUATION',
            integrated_summary_json TEXT NOT NULL DEFAULT '{}',
            model_version TEXT NOT NULL,
            is_demo_record INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY (patient_ref_id) REFERENCES patients(patient_ref_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS vital_signs (
            vital_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL UNIQUE,
            systolic_bp INTEGER,
            diastolic_bp INTEGER,
            heart_rate INTEGER,
            spo2 REAL,
            temperature_c REAL,
            data_source_label TEXT NOT NULL DEFAULT 'USER_REPORTED',
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS fast_assessments (
            fast_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL UNIQUE,
            facial_weakness_reported TEXT NOT NULL DEFAULT 'UNKNOWN',
            facial_ai_score REAL,
            facial_ai_status TEXT NOT NULL DEFAULT 'NOT_PERFORMED',
            facial_metrics_json TEXT NOT NULL DEFAULT '{}',
            arm_weakness_reported TEXT NOT NULL DEFAULT 'UNKNOWN',
            arm_ai_drift_detected TEXT NOT NULL DEFAULT 'NOT_PERFORMED',
            arm_metrics_json TEXT NOT NULL DEFAULT '{}',
            speech_difficulty_reported TEXT NOT NULL DEFAULT 'UNKNOWN',
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS speech_analyses (
            speech_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL UNIQUE,
            transcription TEXT,
            reference_sentence TEXT,
            duration_sec REAL,
            pause_count INTEGER,
            speaking_rate_wpm REAL,
            similarity_score REAL,
            audio_quality_flag TEXT NOT NULL DEFAULT 'NOT_PERFORMED',
            manual_override_status TEXT NOT NULL DEFAULT 'NOT_PERFORMED',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS ct_results (
            ct_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL UNIQUE,
            file_name TEXT,
            modality_format TEXT,
            image_quality_status TEXT NOT NULL DEFAULT 'VALID',
            haemorrhage_pred_json TEXT NOT NULL DEFAULT '{}',
            overall_haemorrhage_prob REAL,
            haemorrhage_detected INTEGER NOT NULL DEFAULT 0,
            uncertainty_entropy REAL,
            ischaemic_seg_status TEXT NOT NULL DEFAULT 'NOT_PERFORMED',
            ischaemic_lesion_burden_pct REAL,
            model_weights_status TEXT NOT NULL,
            gradcam_available INTEGER NOT NULL DEFAULT 0,
            recorded_at TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS generated_reports (
            report_id TEXT PRIMARY KEY,
            assessment_id TEXT NOT NULL,
            pdf_file_path TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            model_version TEXT NOT NULL,
            FOREIGN KEY (assessment_id) REFERENCES assessments(assessment_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS model_metadata (
            model_id TEXT PRIMARY KEY,
            module_name TEXT NOT NULL,
            architecture TEXT NOT NULL,
            dataset_source TEXT NOT NULL,
            weights_loaded INTEGER NOT NULL DEFAULT 0,
            validation_status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
        with self._get_connection() as conn:
            conn.executescript(schema_sql)
            conn.commit()

    def _sync_model_metadata(self) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        ich_loaded = 1 if HAEMORRHAGE_WEIGHTS_PATH.exists() else 0
        isc_loaded = 1 if ISCHAEMIC_WEIGHTS_PATH.exists() else 0

        models = [
            (
                "MOD_CT_ICH_01",
                "CT Haemorrhage Classifier",
                "DenseNet121 (3-Channel HU Windowed)",
                "RSNA Intracranial Hemorrhage Detection Dataset",
                ich_loaded,
                "Trained Checkpoint Loaded" if ich_loaded else "Architecture Ready (Untrained Baseline Weights)",
                now_iso,
            ),
            (
                "MOD_CT_ISC_02",
                "CT Ischaemic Lesion Segmenter",
                "MONAI 2D Residual U-Net",
                "AISD / CPAISD Research Datasets",
                isc_loaded,
                "Trained Checkpoint Loaded" if isc_loaded else "Experimental Architecture Ready (Untrained Baseline Weights)",
                now_iso,
            ),
            (
                "MOD_FACE_03",
                "Facial Asymmetry Landmark Analyzer",
                "MediaPipe Face Mesh / OpenCV Geometric Ratio",
                "Pretrained MediaPipe + Geometric Heuristics",
                1,
                "Experimental Geometric Screening Module",
                now_iso,
            ),
            (
                "MOD_ARM_04",
                "Arm Pronator/Vertical Drift Tracker",
                "MediaPipe Pose + Kinematic Window",
                "Pretrained MediaPipe + Vertical Drift Calculus",
                1,
                "Experimental Kinematic Screening Module",
                now_iso,
            ),
            (
                "MOD_SPEECH_05",
                "Acoustic & Transcription Speech Screener",
                "Faster-Whisper + Librosa Acoustic Feature Pipeline",
                "OpenAI Whisper Pretrained + TORGO Reference Baselines",
                1,
                "Experimental Acoustic & Lexical Screening Module",
                now_iso,
            ),
            (
                "MOD_INT_06",
                "Multimodal Clinical Safety Integration Engine",
                "Deterministic Rule-Based Clinical State Machine",
                "FAST/AHA Stroke Triage Guidelines + Safety Invariants",
                1,
                "Verified Deterministic Safety Layer",
                now_iso,
            ),
        ]

        with self._get_connection() as conn:
            for m in models:
                conn.execute(
                    """
                    INSERT INTO model_metadata (
                        model_id, module_name, architecture, dataset_source,
                        weights_loaded, validation_status, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(model_id) DO UPDATE SET
                        weights_loaded=excluded.weights_loaded,
                        validation_status=excluded.validation_status,
                        updated_at=excluded.updated_at;
                    """,
                    m,
                )
            conn.commit()

    def create_or_update_patient(
        self,
        patient_ref_id: str,
        age: Optional[int],
        symptom_onset_time: Optional[str],
        medical_history: Dict[str, Any],
        is_demo_record: bool = False,
    ) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_id = patient_ref_id.strip().upper()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO patients (
                    patient_ref_id, age, symptom_onset_time,
                    medical_history_json, is_demo_record, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(patient_ref_id) DO UPDATE SET
                    age=excluded.age,
                    symptom_onset_time=excluded.symptom_onset_time,
                    medical_history_json=excluded.medical_history_json,
                    is_demo_record=excluded.is_demo_record;
                """,
                (
                    clean_id,
                    age,
                    symptom_onset_time,
                    json.dumps(medical_history),
                    1 if is_demo_record else 0,
                    now_iso,
                ),
            )
            conn.commit()
        return clean_id

    def create_assessment(
        self,
        patient_ref_id: str,
        urgency_level: str = "PENDING_EVALUATION",
        status: str = "IN_PROGRESS",
        integrated_summary: Optional[Dict[str, Any]] = None,
        is_demo_record: bool = False,
        assessment_id: Optional[str] = None,
    ) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()
        aid = assessment_id or f"ASM-{uuid.uuid4().hex[:8].upper()}"
        summary_json = json.dumps(integrated_summary or {})
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO assessments (
                    assessment_id, patient_ref_id, created_at, status,
                    urgency_level, integrated_summary_json, model_version, is_demo_record
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    status=excluded.status,
                    urgency_level=excluded.urgency_level,
                    integrated_summary_json=excluded.integrated_summary_json;
                """,
                (
                    aid,
                    patient_ref_id.strip().upper(),
                    now_iso,
                    status,
                    urgency_level,
                    summary_json,
                    APP_VERSION,
                    1 if is_demo_record else 0,
                ),
            )
            conn.commit()
        return aid

    def save_vital_signs(
        self,
        assessment_id: str,
        systolic_bp: Optional[int],
        diastolic_bp: Optional[int],
        heart_rate: Optional[int],
        spo2: Optional[float],
        temperature_c: Optional[float],
        data_source_label: str = "USER_REPORTED",
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        vid = f"VIT-{assessment_id}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO vital_signs (
                    vital_id, assessment_id, systolic_bp, diastolic_bp,
                    heart_rate, spo2, temperature_c, data_source_label, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    systolic_bp=excluded.systolic_bp,
                    diastolic_bp=excluded.diastolic_bp,
                    heart_rate=excluded.heart_rate,
                    spo2=excluded.spo2,
                    temperature_c=excluded.temperature_c,
                    data_source_label=excluded.data_source_label,
                    recorded_at=excluded.recorded_at;
                """,
                (
                    vid,
                    assessment_id,
                    systolic_bp,
                    diastolic_bp,
                    heart_rate,
                    spo2,
                    temperature_c,
                    data_source_label,
                    now_iso,
                ),
            )
            conn.commit()

    def save_fast_assessment(
        self,
        assessment_id: str,
        facial_weakness_reported: str,
        facial_ai_score: Optional[float],
        facial_ai_status: str,
        facial_metrics: Dict[str, Any],
        arm_weakness_reported: str,
        arm_ai_drift_detected: str,
        arm_metrics: Dict[str, Any],
        speech_difficulty_reported: str,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        fid = f"FST-{assessment_id}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO fast_assessments (
                    fast_id, assessment_id, facial_weakness_reported,
                    facial_ai_score, facial_ai_status, facial_metrics_json,
                    arm_weakness_reported, arm_ai_drift_detected, arm_metrics_json,
                    speech_difficulty_reported, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    facial_weakness_reported=excluded.facial_weakness_reported,
                    facial_ai_score=excluded.facial_ai_score,
                    facial_ai_status=excluded.facial_ai_status,
                    facial_metrics_json=excluded.facial_metrics_json,
                    arm_weakness_reported=excluded.arm_weakness_reported,
                    arm_ai_drift_detected=excluded.arm_ai_drift_detected,
                    arm_metrics_json=excluded.arm_metrics_json,
                    speech_difficulty_reported=excluded.speech_difficulty_reported,
                    recorded_at=excluded.recorded_at;
                """,
                (
                    fid,
                    assessment_id,
                    facial_weakness_reported,
                    facial_ai_score,
                    facial_ai_status,
                    json.dumps(facial_metrics),
                    arm_weakness_reported,
                    arm_ai_drift_detected,
                    json.dumps(arm_metrics),
                    speech_difficulty_reported,
                    now_iso,
                ),
            )
            conn.commit()

    def save_speech_analysis(
        self,
        assessment_id: str,
        transcription: str,
        reference_sentence: str,
        duration_sec: float,
        pause_count: int,
        speaking_rate_wpm: float,
        similarity_score: float,
        audio_quality_flag: str,
        manual_override_status: str,
        metrics: Dict[str, Any],
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        sid = f"SPC-{assessment_id}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO speech_analyses (
                    speech_id, assessment_id, transcription, reference_sentence,
                    duration_sec, pause_count, speaking_rate_wpm, similarity_score,
                    audio_quality_flag, manual_override_status, metrics_json, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    transcription=excluded.transcription,
                    reference_sentence=excluded.reference_sentence,
                    duration_sec=excluded.duration_sec,
                    pause_count=excluded.pause_count,
                    speaking_rate_wpm=excluded.speaking_rate_wpm,
                    similarity_score=excluded.similarity_score,
                    audio_quality_flag=excluded.audio_quality_flag,
                    manual_override_status=excluded.manual_override_status,
                    metrics_json=excluded.metrics_json,
                    recorded_at=excluded.recorded_at;
                """,
                (
                    sid,
                    assessment_id,
                    transcription,
                    reference_sentence,
                    duration_sec,
                    pause_count,
                    speaking_rate_wpm,
                    similarity_score,
                    audio_quality_flag,
                    manual_override_status,
                    json.dumps(metrics),
                    now_iso,
                ),
            )
            conn.commit()

    def save_ct_result(
        self,
        assessment_id: str,
        file_name: str,
        modality_format: str,
        image_quality_status: str,
        haemorrhage_pred: Dict[str, float],
        overall_haemorrhage_prob: float,
        haemorrhage_detected: bool,
        uncertainty_entropy: float,
        ischaemic_seg_status: str,
        ischaemic_lesion_burden_pct: Optional[float],
        model_weights_status: str,
        gradcam_available: bool,
    ) -> None:
        now_iso = datetime.now(timezone.utc).isoformat()
        cid = f"CT-{assessment_id}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ct_results (
                    ct_id, assessment_id, file_name, modality_format,
                    image_quality_status, haemorrhage_pred_json,
                    overall_haemorrhage_prob, haemorrhage_detected,
                    uncertainty_entropy, ischaemic_seg_status,
                    ischaemic_lesion_burden_pct, model_weights_status,
                    gradcam_available, recorded_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(assessment_id) DO UPDATE SET
                    file_name=excluded.file_name,
                    modality_format=excluded.modality_format,
                    image_quality_status=excluded.image_quality_status,
                    haemorrhage_pred_json=excluded.haemorrhage_pred_json,
                    overall_haemorrhage_prob=excluded.overall_haemorrhage_prob,
                    haemorrhage_detected=excluded.haemorrhage_detected,
                    uncertainty_entropy=excluded.uncertainty_entropy,
                    ischaemic_seg_status=excluded.ischaemic_seg_status,
                    ischaemic_lesion_burden_pct=excluded.ischaemic_lesion_burden_pct,
                    model_weights_status=excluded.model_weights_status,
                    gradcam_available=excluded.gradcam_available,
                    recorded_at=excluded.recorded_at;
                """,
                (
                    cid,
                    assessment_id,
                    file_name,
                    modality_format,
                    image_quality_status,
                    json.dumps(haemorrhage_pred),
                    overall_haemorrhage_prob,
                    1 if haemorrhage_detected else 0,
                    uncertainty_entropy,
                    ischaemic_seg_status,
                    ischaemic_lesion_burden_pct,
                    model_weights_status,
                    1 if gradcam_available else 0,
                    now_iso,
                ),
            )
            conn.commit()

    def record_generated_report(self, assessment_id: str, pdf_file_path: str) -> str:
        now_iso = datetime.now(timezone.utc).isoformat()
        rid = f"RPT-{uuid.uuid4().hex[:8].upper()}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO generated_reports (
                    report_id, assessment_id, pdf_file_path, generated_at, model_version
                ) VALUES (?, ?, ?, ?, ?);
                """,
                (rid, assessment_id, str(pdf_file_path), now_iso, APP_VERSION),
            )
            conn.commit()
        return rid

    def get_full_assessment_bundle(self, assessment_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the complete joined clinical record for an assessment."""
        with self._get_connection() as conn:
            asm = conn.execute(
                "SELECT * FROM assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            if not asm:
                return None
            asm_dict = dict(asm)
            pat = conn.execute(
                "SELECT * FROM patients WHERE patient_ref_id = ?",
                (asm_dict["patient_ref_id"],),
            ).fetchone()
            vit = conn.execute(
                "SELECT * FROM vital_signs WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            fast = conn.execute(
                "SELECT * FROM fast_assessments WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            spc = conn.execute(
                "SELECT * FROM speech_analyses WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()
            ct = conn.execute(
                "SELECT * FROM ct_results WHERE assessment_id = ?", (assessment_id,)
            ).fetchone()

            return {
                "assessment": asm_dict,
                "patient": dict(pat) if pat else {},
                "vitals": dict(vit) if vit else {},
                "fast": dict(fast) if fast else {},
                "speech": dict(spc) if spc else {},
                "ct": dict(ct) if ct else {},
            }

    def get_dashboard_metrics(self) -> Dict[str, Any]:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]
            demo_count = conn.execute(
                "SELECT COUNT(*) FROM assessments WHERE is_demo_record = 1"
            ).fetchone()[0]
            urgent = conn.execute(
                """
                SELECT COUNT(*) FROM assessments
                WHERE urgency_level IN ('IMMEDIATE_EMERGENCY_ESCALATION', 'HIGH_PRIORITY_CLINICAL_REVIEW')
                """
            ).fetchone()[0]
            ct_count = conn.execute("SELECT COUNT(*) FROM ct_results").fetchone()[0]
            ich_pos = conn.execute(
                "SELECT COUNT(*) FROM ct_results WHERE haemorrhage_detected = 1"
            ).fetchone()[0]

        return {
            "total_assessments": total,
            "demo_assessments": demo_count,
            "live_assessments": total - demo_count,
            "urgent_assessments": urgent,
            "ct_scans_analysed": ct_count,
            "haemorrhage_positive_count": ich_pos,
        }

    def get_recent_assessments(self, limit: int = 50) -> List[Dict[str, Any]]:
        query = """
        SELECT
            a.assessment_id,
            a.patient_ref_id,
            a.created_at,
            a.status,
            a.urgency_level,
            a.is_demo_record,
            a.model_version,
            p.age,
            p.symptom_onset_time,
            f.facial_weakness_reported,
            f.arm_weakness_reported,
            f.speech_difficulty_reported,
            c.file_name AS ct_file,
            c.overall_haemorrhage_prob,
            c.haemorrhage_detected,
            c.image_quality_status
        FROM assessments a
        LEFT JOIN patients p ON a.patient_ref_id = p.patient_ref_id
        LEFT JOIN fast_assessments f ON a.assessment_id = f.assessment_id
        LEFT JOIN ct_results c ON a.assessment_id = c.assessment_id
        ORDER BY a.created_at DESC
        LIMIT ?;
        """
        with self._get_connection() as conn:
            rows = conn.execute(query, (limit,)).fetchall()
            return [dict(r) for r in rows]

    def get_model_metadata(self) -> List[Dict[str, Any]]:
        self._sync_model_metadata()
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM model_metadata ORDER BY model_id ASC"
            ).fetchall()
            return [dict(r) for r in rows]

    def seed_demo_scenarios(self) -> int:
        # Scenario A: Haemorrhage-positive CT case with acute symptoms
        self.create_or_update_patient(
            patient_ref_id="DEMO-SCENARIO-A",
            age=68,
            symptom_onset_time="45 minutes ago",
            medical_history={"hypertension": True, "anticoagulants": True, "diabetes": False},
            is_demo_record=True,
        )
        aid_a = self.create_assessment(
            patient_ref_id="DEMO-SCENARIO-A",
            urgency_level="IMMEDIATE_EMERGENCY_ESCALATION",
            status="COMPLETED",
            integrated_summary={
                "scenario": "Scenario A: Acute Haemorrhage-Positive Case",
                "urgency_level": "IMMEDIATE_EMERGENCY_ESCALATION",
                "escalation_reasons": [
                    "Suspected Intraparenchymal & Intraventricular Haemorrhage on Brain CT",
                    "Acute Right Facial Asymmetry & Left Arm Drift Reported",
                    "Severe Hypertensive Presentation (188/106 mmHg) [SIMULATED_DEMO]",
                ],
                "conflicts": [],
                "missing_information": [],
                "recommended_action": "IMMEDIATE EMERGENCY ESCALATION: Activate emergency stroke & neurosurgical protocol immediately.",
            },
            is_demo_record=True,
            assessment_id="ASM-DEMOA001",
        )
        self.save_vital_signs(aid_a, 188, 106, 94, 96.0, 37.1, "SIMULATED_DEMO")
        self.save_fast_assessment(
            aid_a,
            facial_weakness_reported="YES",
            facial_ai_score=0.28,
            facial_ai_status="ASYMMETRY_FLAGGED",
            facial_metrics={"mouth_angle_deg": 6.4, "source": "SIMULATED_DEMO"},
            arm_weakness_reported="YES",
            arm_ai_drift_detected="DRIFT_FLAGGED",
            arm_metrics={"left_wrist_drop_norm": 0.19, "source": "SIMULATED_DEMO"},
            speech_difficulty_reported="YES",
        )
        self.save_ct_result(
            aid_a,
            file_name="demo_scenario_a_ich_positive.dcm",
            modality_format="DICOM",
            image_quality_status="VALID",
            haemorrhage_pred={
                "intraparenchymal": 0.91,
                "intraventricular": 0.64,
                "subarachnoid": 0.12,
                "subdural": 0.08,
                "epidural": 0.02,
                "any_haemorrhage": 0.94,
            },
            overall_haemorrhage_prob=0.94,
            haemorrhage_detected=True,
            uncertainty_entropy=0.18,
            ischaemic_seg_status="NOT_INDICATED_PRIMARY_ICH",
            ischaemic_lesion_burden_pct=0.0,
            model_weights_status="DEMO_REFERENCE_RECORD",
            gradcam_available=True,
        )

        # Scenario B: Stroke-like symptoms (FAST Positive) but Negative CT Haemorrhage
        self.create_or_update_patient(
            patient_ref_id="DEMO-SCENARIO-B",
            age=61,
            symptom_onset_time="90 minutes ago",
            medical_history={"atrial_fibrillation": True, "hypertension": True},
            is_demo_record=True,
        )
        aid_b = self.create_assessment(
            patient_ref_id="DEMO-SCENARIO-B",
            urgency_level="IMMEDIATE_EMERGENCY_ESCALATION",
            status="COMPLETED",
            integrated_summary={
                "scenario": "Scenario B: FAST-Positive / CT Haemorrhage-Negative (Suspected Acute Ischaemia Safety Rule)",
                "urgency_level": "IMMEDIATE_EMERGENCY_ESCALATION",
                "escalation_reasons": [
                    "CRITICAL SAFETY OVERRIDE: Sudden Facial Weakness, Arm Drift, and Dysarthric Speech reported.",
                    "CT Haemorrhage is negative (prob 0.06), which NEVER rules out acute ischaemic stroke.",
                ],
                "conflicts": [
                    "Clinical FAST symptoms are strongly positive despite negative CT haemorrhage classification."
                ],
                "missing_information": [],
                "recommended_action": "IMMEDIATE EMERGENCY ESCALATION: Negative CT haemorrhage does NOT rule out ischaemic stroke. Evaluate immediately for acute thrombolysis/thrombectomy pathway.",
            },
            is_demo_record=True,
            assessment_id="ASM-DEMOB002",
        )
        self.save_vital_signs(aid_b, 134, 82, 88, 98.0, 36.8, "SIMULATED_DEMO")
        self.save_fast_assessment(
            aid_b,
            facial_weakness_reported="YES",
            facial_ai_score=0.24,
            facial_ai_status="ASYMMETRY_FLAGGED",
            facial_metrics={"mouth_angle_deg": 5.1, "source": "SIMULATED_DEMO"},
            arm_weakness_reported="YES",
            arm_ai_drift_detected="DRIFT_FLAGGED",
            arm_metrics={"right_wrist_drop_norm": 0.22, "source": "SIMULATED_DEMO"},
            speech_difficulty_reported="YES",
        )
        self.save_ct_result(
            aid_b,
            file_name="demo_scenario_b_ncct_negative_ich.dcm",
            modality_format="DICOM",
            image_quality_status="VALID",
            haemorrhage_pred={
                "intraparenchymal": 0.03,
                "intraventricular": 0.01,
                "subarachnoid": 0.02,
                "subdural": 0.04,
                "epidural": 0.01,
                "any_haemorrhage": 0.06,
            },
            overall_haemorrhage_prob=0.06,
            haemorrhage_detected=False,
            uncertainty_entropy=0.14,
            ischaemic_seg_status="SUSPECTED_EARLY_MCA_TERRITORY",
            ischaemic_lesion_burden_pct=3.8,
            model_weights_status="DEMO_REFERENCE_RECORD",
            gradcam_available=True,
        )

        # Scenario C: Missing CT and Low-Quality/Incomplete Data with Unclear Onset
        self.create_or_update_patient(
            patient_ref_id="DEMO-SCENARIO-C",
            age=74,
            symptom_onset_time="Unknown (Wake-up presentation)",
            medical_history={"prior_tia": True},
            is_demo_record=True,
        )
        aid_c = self.create_assessment(
            patient_ref_id="DEMO-SCENARIO-C",
            urgency_level="IMMEDIATE_EMERGENCY_ESCALATION",
            status="INCOMPLETE_DATA_ESCALATED",
            integrated_summary={
                "scenario": "Scenario C: Missing CT Scan & Poor Sensor Quality with Manual Symptom Escalation",
                "urgency_level": "IMMEDIATE_EMERGENCY_ESCALATION",
                "escalation_reasons": [
                    "Sudden Speech Difficulty reported by caregiver.",
                    "Missing Brain CT scan and unavailable webcam lighting do NOT delay emergency triage.",
                ],
                "conflicts": ["Webcam facial analysis failed due to low lighting; manual observation used."],
                "missing_information": ["Brain CT Scan", "Blood Pressure", "Body Temperature"],
                "recommended_action": "IMMEDIATE EMERGENCY ESCALATION: Do not wait for missing CT or camera sensors when sudden speech impairment is reported.",
            },
            is_demo_record=True,
            assessment_id="ASM-DEMOC003",
        )
        self.save_vital_signs(aid_c, None, None, 76, 97.0, None, "SIMULATED_DEMO")
        self.save_fast_assessment(
            aid_c,
            facial_weakness_reported="UNSURE",
            facial_ai_score=None,
            facial_ai_status="POOR_LIGHTING_REJECTED",
            facial_metrics={"error": "Ambient luminance below threshold", "source": "SIMULATED_DEMO"},
            arm_weakness_reported="NO",
            arm_ai_drift_detected="NOT_PERFORMED",
            arm_metrics={},
            speech_difficulty_reported="YES",
        )
        return 3

    def delete_demo_records(self) -> int:
        with self._get_connection() as conn:
            cur = conn.execute("DELETE FROM patients WHERE is_demo_record = 1;")
            conn.execute("DELETE FROM assessments WHERE is_demo_record = 1;")
            conn.commit()
            return cur.rowcount
